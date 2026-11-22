// mend — one patch tool for every AI edit format.
//
// Per trial the harness runs `node main.ts` with:
//   MAKEFUN_PROMPT_FILE  path to the trial brief (a patch task, in any
//                        common AI edit format, against a shop simulator)
//   MAKEFUN_SHOP_GRAPHQL_URL / MAKEFUN_SHOP_TOKEN  Shopify Admin GraphQL
//   MAKEFUN_SUPPLIER_URL   supplier REST endpoint
//   ANTHROPIC_BASE_URL / OPENAI_BASE_URL  LLM gateway forwarder
//   MAKEFUN_TRIAL_SEED / MAKEFUN_TRIAL_NOW
//
// Design: deterministic code does the parsing, execution and verification;
// the model only grounds candidate edits onto real shop entities and
// assembles the mutation plan. Sequential awaits, temperature 0, no
// randomness, no wall-clock timestamps inside any request, so a replay
// with the same brief + model responses + env responses is byte-identical.
// Always exits 0; partial state is still scored.

import * as fs from "node:fs";

const START_MS = Date.now();
const DEADLINE_MS = 150_000; // internal deadline, wall budget is 180s
const MAX_MODEL_CALLS = 4;
const MODEL_TIMEOUT_MS = 30_000;
const SHOP_TIMEOUT_MS = 20_000;

function timeLeftMs(): number {
  return DEADLINE_MS - (Date.now() - START_MS);
}
function env(name: string): string {
  return process.env[name] ?? "";
}

// ---------------------------------------------------------------- types
type Candidate = {
  format: string;
  target: string;
  field: string;
  oldText: string;
  newText: string;
};

type Op = {
  kind: string;
  mutation?: string;
  id?: string;
  productId?: string;
  fields?: Record<string, unknown>;
  variants?: Array<Record<string, unknown>>;
  method?: string;
  path?: string;
  body?: unknown;
};

type Product = {
  id: string;
  title: string;
  handle: string;
  status: string;
  vendor: string;
  productType: string;
  tags: string[];
  description: string;
  variants: Array<{ id: string; price: string; sku: string; inventoryQuantity: number | null }>;
};

// ---------------------------------------------------------------- fetch
async function fetchJson(url: string, init: Record<string, unknown>, timeoutMs: number): Promise<{ ok: boolean; status: number; json: unknown; text: string }> {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...(init as object), signal: ctrl.signal } as RequestInit);
    const text = await res.text();
    let json: unknown = null;
    try { json = JSON.parse(text); } catch { /* not json */ }
    return { ok: res.ok, status: res.status, json, text };
  } catch (e) {
    return { ok: false, status: 0, json: null, text: String(e) };
  } finally {
    clearTimeout(t);
  }
}

function truncate(s: string, n: number): string {
  return s.length > n ? s.slice(0, n) + "\n…[truncated]" : s;
}

// ---------------------------------------------------------------- field inference
const FIELD_WORDS: Array<[RegExp, string]> = [
  [/title|judul/i, "title"],
  [/description|deskripsi/i, "description"],
  [/price|harga/i, "price"],
  [/tags?|label/i, "tags"],
  [/status/i, "status"],
  [/vendor|brand|merek/i, "vendor"],
  [/product.?type|tipe|kategori/i, "productType"],
  [/sku/i, "sku"],
  [/stock|stok|inventory|quantity|qty/i, "inventory"],
  [/handle|slug/i, "handle"],
];

function inferField(context: string): string {
  for (const [re, field] of FIELD_WORDS) {
    if (re.test(context)) return field;
  }
  return "";
}

function cleanLine(s: string): string {
  return s.replace(/^\s*[`"'“”']+|[`"'“”']+\s*$/g, "").trim();
}

// ---------------------------------------------------------------- parsers (deterministic, no model)
function parseUnifiedDiff(text: string): Candidate[] {
  const out: Candidate[] = [];
  const lines = text.split("\n");
  let file = "";
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    const plus = line.match(/^\+\+\+\s+(?:b\/)?(.+?)\s*$/);
    if (plus) file = plus[1].trim();
    const hunk = line.match(/^@@.*@@\s*(.*)$/);
    if (hunk) {
      const ctx = (hunk[1] || "") + " " + file;
      const removed: string[] = [];
      const added: string[] = [];
      i++;
      while (i < lines.length && /^[-+ \\]/.test(lines[i]) && !lines[i].startsWith("---") && !lines[i].startsWith("+++")) {
        if (lines[i].startsWith("-")) removed.push(lines[i].slice(1));
        else if (lines[i].startsWith("+")) added.push(lines[i].slice(1));
        i++;
      }
      if (removed.length || added.length) {
        out.push({
          format: "unified-diff",
          target: file,
          field: inferField(ctx + "\n" + removed.join(" ") + " " + added.join(" ")),
          oldText: removed.join("\n"),
          newText: added.join("\n"),
        });
      }
      continue;
    }
    i++;
  }
  return out;
}

function parseSearchReplace(text: string): Candidate[] {
  const out: Candidate[] = [];
  const patterns: RegExp[] = [
    /<<<<<<< SEARCH\s*\n([\s\S]*?)\n={3,}\s*\n([\s\S]*?)\n>>>>>>> REPLACE/g,
    /<<<<<<< SEARCH\s*\n([\s\S]*?)\n={3,}\s*\n([\s\S]*?)\n>>>>>>>/g,
    /```search\s*\n([\s\S]*?)\n```\s*\n```replace\s*\n([\s\S]*?)\n```/gi,
    /---+\s*SEARCH\s*-*\s*\n([\s\S]*?)\n---+\s*REPLACE\s*-*\s*\n([\s\S]*?)(?=\n---|\n\s*$|\n[A-Z][A-Za-z ]{0,30}:\s*\n)/g,
  ];
  for (const re of patterns) {
    let m: RegExpExecArray | null;
    while ((m = re.exec(text)) !== null) {
      const oldText = m[1].trim();
      const newText = m[2].trim();
      if (!oldText && !newText) continue;
      out.push({
        format: "search-replace",
        target: "",
        field: inferField(oldText + " " + newText),
        oldText,
        newText,
      });
    }
  }
  return out;
}

function parseBeginPatch(text: string): Candidate[] {
  const out: Candidate[] = [];
  const block = text.match(/\*\*\*\s*Begin Patch\s*\*\*\*\s*\n([\s\S]*?)\n\*\*\*\s*End Patch\s*\*\*\*/);
  if (!block) return out;
  const body = block[1];
  const sections = body.split(/^\*\*\*\s*Update File:\s*(.+?)\s*$/m);
  for (let s = 1; s + 1 < sections.length; s += 2) {
    const file = sections[s].trim();
    const chunk = sections[s + 1];
    const removed: string[] = [];
    const added: string[] = [];
    for (const line of chunk.split("\n")) {
      if (line.startsWith("-") && !line.startsWith("---")) removed.push(line.slice(1));
      else if (line.startsWith("+") && !line.startsWith("+++")) added.push(line.slice(1));
    }
    if (removed.length || added.length) {
      out.push({
        format: "begin-patch",
        target: file,
        field: inferField(file + "\n" + removed.join(" ") + " " + added.join(" ")),
        oldText: removed.join("\n"),
        newText: added.join("\n"),
      });
    }
  }
  if (out.length === 0) {
    // No per-file sections: treat whole block as one patch.
    const removed: string[] = [];
    const added: string[] = [];
    for (const line of body.split("\n")) {
      if (line.startsWith("-") && !line.startsWith("---")) removed.push(line.slice(1));
      else if (line.startsWith("+") && !line.startsWith("+++")) added.push(line.slice(1));
    }
    if (removed.length || added.length) {
      out.push({ format: "begin-patch", target: "", field: inferField(removed.join(" ") + " " + added.join(" ")), oldText: removed.join("\n"), newText: added.join("\n") });
    }
  }
  return out;
}

function parseOldNew(text: string): Candidate[] {
  const out: Candidate[] = [];
  const seen = new Set<string>();
  const push = (c: Candidate) => {
    const key = c.oldText + "\u0000" + c.newText;
    if ((c.oldText || c.newText) && !seen.has(key)) {
      seen.add(key);
      out.push(c);
    }
  };
  // Labeled blocks: old_text/new_text, old/new, before/after, with optional file label.
  const labelRe = /(?:file|path|target)\s*[:=]\s*([^\n]+)\n[\s\S]{0,200}?(?:old_text|old|before)\s*[:=]\s*(?:"([^"\n]*)"|'([^'\n]*)'|([^\n]+))\n(?:new_text|new|after)\s*[:=]\s*(?:"([^"\n]*)"|'([^'\n]*)'|([^\n]+))/gi;
  let m: RegExpExecArray | null;
  while ((m = labelRe.exec(text)) !== null) {
    const target = cleanLine(m[1] || "");
    const oldText = (m[2] ?? m[3] ?? m[4] ?? "").trim();
    const newText = (m[5] ?? m[6] ?? m[7] ?? "").trim();
    push({ format: "old-new", target, field: inferField(target + " " + oldText + " " + newText), oldText, newText });
  }
  const labelRe2 = /(?:old_text|old|before)\s*[:=]\s*(?:"([^"\n]*)"|'([^'\n]*)'|([^\n]+))\n(?:new_text|new|after)\s*[:=]\s*(?:"([^"\n]*)"|'([^'\n]*)'|([^\n]+))/gi;
  while ((m = labelRe2.exec(text)) !== null) {
    const oldText = (m[1] ?? m[2] ?? m[3] ?? "").trim();
    const newText = (m[4] ?? m[5] ?? m[6] ?? "").trim();
    push({ format: "old-new", target: "", field: inferField(oldText + " " + newText), oldText, newText });
  }
  // Fenced old/new code blocks back to back.
  const fenceRe = /```(?:old|before)[^\n]*\n([\s\S]*?)\n```\s*\n```(?:new|after)[^\n]*\n([\s\S]*?)\n```/gi;
  while ((m = fenceRe.exec(text)) !== null) {
    push({ format: "old-new", target: "", field: inferField(m[1] + " " + m[2]), oldText: m[1].trim(), newText: m[2].trim() });
  }
  return out;
}

function parseArrow(text: string): Candidate[] {
  const out: Candidate[] = [];
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line || line.length > 400) continue;
    // "ganti X menjadi Y" / "ubah X ke Y" / "change X to Y"
    let m = line.match(/^(?:ganti|ubah|change|replace)\s+(.+?)\s+(?:menjadi|ke|to|dengan)\s+(.+?)\s*$/i);
    if (m) {
      const field = inferField(m[1]);
      out.push({ format: "arrow", target: "", field, oldText: cleanLine(m[1].replace(new RegExp("^" + field + "\\s*[:\\-]?\\s*", "i"), "")), newText: cleanLine(m[2]) });
      continue;
    }
    // "field: old -> new" or "X -> Y" (require -> and a quoted or field-like left side)
    m = line.match(/^(.{1,120}?)\s*->\s*(.+?)\s*$/);
    if (m && (m[1].includes('"') || m[1].includes("'") || /^(title|description|price|tags?|status|vendor|sku|handle|type)\b/i.test(m[1]))) {
      const left = cleanLine(m[1]);
      const fieldMatch = left.match(/^([A-Za-z][A-Za-z ]{0,30}?)\s*[:\-]\s*(.+)$/);
      const field = inferField(left);
      out.push({
        format: "arrow",
        target: "",
        field,
        oldText: fieldMatch ? cleanLine(fieldMatch[2]) : left,
        newText: cleanLine(m[2]),
      });
    }
  }
  return out;
}

function walkJson(v: unknown, out: Candidate[]): void {
  if (Array.isArray(v)) {
    for (const item of v) walkJson(item, out);
    return;
  }
  if (v && typeof v === "object") {
    const o = v as Record<string, unknown>;
    const str = (x: unknown) => (typeof x === "string" ? x : x == null ? "" : JSON.stringify(x));
    // JSON Patch style: {op, path, value} / {op, path, from, to}
    if (typeof o.op === "string" && typeof o.path === "string") {
      const field = inferField(o.path) || o.path.split("/").filter(Boolean).pop() || "";
      out.push({
        format: "json-op",
        target: o.path,
        field,
        oldText: str(o.from ?? o.old ?? ""),
        newText: str(o.value ?? o.to ?? o.new ?? ""),
      });
      return;
    }
    // {field, old, new} / {field, from, to} / {field, value}
    if (typeof o.field === "string" && (o.old !== undefined || o.new !== undefined || o.from !== undefined || o.to !== undefined || o.value !== undefined)) {
      out.push({
        format: "json-op",
        target: str(o.target ?? o.product ?? o.handle ?? ""),
        field: inferField(o.field) || o.field,
        oldText: str(o.old ?? o.from ?? ""),
        newText: str(o.new ?? o.to ?? o.value ?? ""),
      });
      return;
    }
    // {old, new} pair anywhere
    if (o.old !== undefined && o.new !== undefined) {
      out.push({ format: "json-op", target: str(o.target ?? ""), field: inferField(JSON.stringify(o)), oldText: str(o.old), newText: str(o.new) });
      return;
    }
    for (const v2 of Object.values(o)) walkJson(v2, out);
  }
}

function parseJsonOps(text: string): Candidate[] {
  const out: Candidate[] = [];
  // Find balanced {...} / [...] spans and try JSON.parse on each.
  const spans: string[] = [];
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (ch !== "{" && ch !== "[") continue;
    const open = ch;
    const close = ch === "{" ? "}" : "]";
    let depth = 0;
    let inStr = false;
    let esc = false;
    for (let j = i; j < text.length && j < i + 20000; j++) {
      const c = text[j];
      if (inStr) {
        if (esc) esc = false;
        else if (c === "\\") esc = true;
        else if (c === '"') inStr = false;
      } else if (c === '"') inStr = true;
      else if (c === open) depth++;
      else if (c === close) {
        depth--;
        if (depth === 0) {
          spans.push(text.slice(i, j + 1));
          i = j;
          break;
        }
      }
    }
  }
  for (const span of spans) {
    try {
      walkJson(JSON.parse(span), out);
    } catch { /* not JSON */ }
  }
  return out;
}

function parseCandidates(brief: string): Candidate[] {
  const all = [
    ...parseUnifiedDiff(brief),
    ...parseSearchReplace(brief),
    ...parseBeginPatch(brief),
    ...parseOldNew(brief),
    ...parseArrow(brief),
    ...parseJsonOps(brief),
  ];
  // Deduplicate identical (old,new) pairs, keep first format.
  const seen = new Set<string>();
  const out: Candidate[] = [];
  for (const c of all) {
    const key = c.oldText + "\u0000" + c.newText;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(c);
  }
  return out.slice(0, 30);
}

// ---------------------------------------------------------------- shop (GraphQL)
function shopHeaders(): Record<string, string> {
  const h: Record<string, string> = { "Content-Type": "application/json" };
  const tok = env("MAKEFUN_SHOP_TOKEN");
  if (tok) h["X-Shopify-Access-Token"] = tok;
  return h;
}

async function gql(query: string, variables?: Record<string, unknown>): Promise<{ data: unknown; errors: unknown }> {
  const url = env("MAKEFUN_SHOP_GRAPHQL_URL");
  if (!url) return { data: null, errors: "no shop url" };
  const r = await fetchJson(url, {
    method: "POST",
    headers: shopHeaders(),
    body: JSON.stringify({ query, variables: variables ?? {} }),
  }, SHOP_TIMEOUT_MS);
  if (!r.ok || !r.json || typeof r.json !== "object") return { data: null, errors: r.text.slice(0, 300) };
  const j = r.json as Record<string, unknown>;
  return { data: j.data ?? null, errors: j.errors ?? null };
}

async function introspectShop(): Promise<string> {
  const q = `{ __schema { queryType { fields { name } } mutationType { fields { name } } } }`;
  const r = await gql(q);
  if (!r.data) return "introspection unavailable";
  const d = r.data as { __schema?: { queryType?: { fields?: Array<{ name: string }> }; mutationType?: { fields?: Array<{ name: string }> } } };
  const qf = (d.__schema?.queryType?.fields ?? []).map((f) => f.name).slice(0, 60);
  const mf = (d.__schema?.mutationType?.fields ?? []).map((f) => f.name).slice(0, 80);
  return "Query fields: " + qf.join(", ") + "\nMutation fields: " + mf.join(", ");
}

const PRODUCTS_QUERY = `query { products(first: 50) { edges { node {
  id title handle status vendor productType tags description
  variants(first: 10) { edges { node { id price sku inventoryQuantity } } }
} } } }`;

function normalizeProducts(data: unknown): Product[] {
  const out: Product[] = [];
  const d = data as { products?: { edges?: Array<{ node: Record<string, unknown> }> } };
  for (const e of d?.products?.edges ?? []) {
    const n = e.node as Record<string, unknown>;
    const variants: Product["variants"] = [];
    const vedges = (n.variants as { edges?: Array<{ node: Record<string, unknown> }> } | undefined)?.edges ?? [];
    for (const ve of vedges) {
      const vn = ve.node;
      variants.push({
        id: String(vn.id ?? ""),
        price: String(vn.price ?? ""),
        sku: String(vn.sku ?? ""),
        inventoryQuantity: typeof vn.inventoryQuantity === "number" ? vn.inventoryQuantity : null,
      });
    }
    out.push({
      id: String(n.id ?? ""),
      title: String(n.title ?? ""),
      handle: String(n.handle ?? ""),
      status: String(n.status ?? ""),
      vendor: String(n.vendor ?? ""),
      productType: String(n.productType ?? ""),
      tags: Array.isArray(n.tags) ? (n.tags as unknown[]).map(String) : [],
      description: String(n.description ?? ""),
      variants,
    });
  }
  return out;
}

async function fetchProducts(): Promise<Product[]> {
  const r = await gql(PRODUCTS_QUERY);
  if (!r.data) return [];
  return normalizeProducts(r.data);
}

async function fetchExtraEntities(brief: string): Promise<string> {
  const low = brief.toLowerCase();
  const parts: string[] = [];
  const want = (words: string[]) => words.some((w) => low.includes(w));
  if (want(["collection"])) {
    const r = await gql(`query { collections(first: 20) { edges { node { id title handle } } } }`);
    if (r.data) parts.push("collections: " + truncate(JSON.stringify(r.data), 2000));
  }
  if (want(["order"])) {
    const r = await gql(`query { orders(first: 20) { edges { node { id name } } } }`);
    if (r.data) parts.push("orders: " + truncate(JSON.stringify(r.data), 2000));
  }
  if (want(["customer"])) {
    const r = await gql(`query { customers(first: 20) { edges { node { id displayName email } } } }`);
    if (r.data) parts.push("customers: " + truncate(JSON.stringify(r.data), 2000));
  }
  if (want(["discount", "coupon", "code"])) {
    const r = await gql(`query { codeDiscountNodes(first: 20) { edges { node { id } } } }`);
    if (r.data) parts.push("discounts: " + truncate(JSON.stringify(r.data), 1000));
  }
  return parts.join("\n");
}

// ---------------------------------------------------------------- model
let modelCalls = 0;

async function listModels(): Promise<{ base: string; kind: string; models: string[] }> {
  const anthropicBase = env("ANTHROPIC_BASE_URL");
  const openaiBase = env("OPENAI_BASE_URL");
  const key = env("ANTHROPIC_API_KEY") || env("OPENAI_API_KEY") || "makefun-trial";
  if (anthropicBase) {
    const r = await fetchJson(anthropicBase.replace(/\/$/, "") + "/v1/models", {
      headers: { "x-api-key": key, "anthropic-version": "2023-06-01" },
    }, MODEL_TIMEOUT_MS);
    const j = r.json as { data?: Array<{ id?: string }> } | null;
    if (r.ok && j?.data?.length) {
      return { base: anthropicBase, kind: "anthropic", models: j.data.map((m) => m.id ?? "").filter(Boolean) };
    }
  }
  if (openaiBase) {
    const r = await fetchJson(openaiBase.replace(/\/$/, "") + "/v1/models", {
      headers: { Authorization: "Bearer " + (env("OPENAI_API_KEY") || key) },
    }, MODEL_TIMEOUT_MS);
    const j = r.json as { data?: Array<{ id?: string }> } | null;
    if (r.ok && j?.data?.length) {
      return { base: openaiBase, kind: "openai", models: j.data.map((m) => m.id ?? "").filter(Boolean) };
    }
  }
  return { base: anthropicBase || openaiBase, kind: anthropicBase ? "anthropic" : "openai", models: [] };
}

function extractJsonObject(text: string): unknown {
  const start = text.indexOf("{");
  if (start < 0) return null;
  let depth = 0;
  let inStr = false;
  let esc = false;
  for (let j = start; j < text.length; j++) {
    const c = text[j];
    if (inStr) {
      if (esc) esc = false;
      else if (c === "\\") esc = true;
      else if (c === '"') inStr = false;
    } else if (c === '"') inStr = true;
    else if (c === "{") depth++;
    else if (c === "}") {
      depth--;
      if (depth === 0) {
        try { return JSON.parse(text.slice(start, j + 1)); } catch { return null; }
      }
    }
  }
  return null;
}

async function callModel(base: string, kind: string, model: string, system: string, userText: string): Promise<string> {
  modelCalls++;
  const key = env("ANTHROPIC_API_KEY") || env("OPENAI_API_KEY") || "makefun-trial";
  const url = base.replace(/\/$/, "") + (kind === "anthropic" ? "/v1/messages" : "/v1/chat/completions");
  const body = kind === "anthropic"
    ? { model, max_tokens: 2048, temperature: 0, system, messages: [{ role: "user", content: userText }] }
    : { model, max_tokens: 2048, temperature: 0, messages: [{ role: "system", content: system }, { role: "user", content: userText }] };
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (kind === "anthropic") {
    headers["x-api-key"] = key;
    headers["anthropic-version"] = "2023-06-01";
  } else {
    headers["Authorization"] = "Bearer " + (env("OPENAI_API_KEY") || key);
  }
  const r = await fetchJson(url, { method: "POST", headers, body: JSON.stringify(body) }, MODEL_TIMEOUT_MS);
  if (!r.ok || !r.json || typeof r.json !== "object") return "";
  const j = r.json as Record<string, unknown>;
  if (kind === "anthropic") {
    const content = j.content as Array<{ text?: string }> | undefined;
    return (content ?? []).map((c) => c.text ?? "").join("\n");
  }
  const choices = j.choices as Array<{ message?: { content?: string } }> | undefined;
  return choices?.[0]?.message?.content ?? "";
}

const PLAN_SYSTEM = "You are mend, a patch agent for a shop. You receive a brief describing changes to shop data in some AI edit format, a compact shop schema, the current shop snapshot, and deterministic candidate edits parsed from the brief. Reply with exactly one JSON object and nothing else: {\"operations\": [...]}. Each operation is one of: {\"kind\":\"graphql\",\"mutation\":\"productUpdate\",\"id\":\"gid://shopify/Product/…\",\"fields\":{\"title\":\"…\",\"description\":\"…\",\"tags\":[\"…\"],\"status\":\"ACTIVE|DRAFT|ARCHIVED\",\"vendor\":\"…\",\"productType\":\"…\",\"handle\":\"…\"}} | {\"kind\":\"graphql\",\"mutation\":\"productVariantsBulkUpdate\",\"productId\":\"gid://…\",\"variants\":[{\"id\":\"gid://shopify/ProductVariant/…\",\"price\":\"12.99\",\"sku\":\"…\"}]} | {\"kind\":\"graphql\",\"mutation\":\"productCreate\",\"fields\":{…same product fields…}} | {\"kind\":\"graphql\",\"mutation\":\"productDelete\",\"id\":\"gid://…\"} | {\"kind\":\"graphql\",\"mutation\":\"collectionUpdate\",\"id\":\"gid://…\",\"fields\":{\"title\":\"…\",\"description\":\"…\"}} | {\"kind\":\"supplier\",\"method\":\"GET|POST\",\"path\":\"/…\",\"body\":{…}}. Ground every id/title/price to the snapshot; use candidate edits as the source of old/new text; never invent products, ids or prices; if a requested change cannot be grounded, omit it. Keep the plan minimal: only the operations the brief asks for.";

function planPrompt(brief: string, schema: string, snapshot: string, candidates: Candidate[], extra: string): string {
  return [
    "BRIEF:",
    truncate(brief, 12000),
    "",
    "SHOP SCHEMA:",
    truncate(schema, 3000),
    "",
    "CURRENT SHOP SNAPSHOT (products):",
    truncate(snapshot, 8000),
    extra ? "\nOTHER ENTITIES:\n" + truncate(extra, 3000) : "",
    "",
    "PARSED CANDIDATE EDITS (deterministic parsers; format/target/field/old/new):",
    truncate(JSON.stringify(candidates, null, 1), 6000),
    "",
    "TRIAL: seed=" + env("MAKEFUN_TRIAL_SEED") + " now=" + env("MAKEFUN_TRIAL_NOW"),
    "",
    "Return exactly one JSON object: {\"operations\": [...]} as specified in the system prompt.",
  ].join("\n");
}

// ---------------------------------------------------------------- execution
const ALLOWED_MUTATIONS = new Set([
  "productUpdate",
  "productVariantsBulkUpdate",
  "productCreate",
  "productDelete",
  "collectionUpdate",
]);

function buildMutation(op: Op): { query: string; variables: Record<string, unknown> } | null {
  if (op.mutation === "productUpdate" && op.id) {
    const input: Record<string, unknown> = { id: op.id };
    const f = op.fields ?? {};
    if (typeof f.title === "string") input.title = f.title;
    if (typeof f.description === "string") input.descriptionHtml = f.description;
    if (typeof f.descriptionHtml === "string") input.descriptionHtml = f.descriptionHtml;
    if (Array.isArray(f.tags)) input.tags = f.tags;
    else if (typeof f.tags === "string") input.tags = f.tags.split(",").map((t) => t.trim()).filter(Boolean);
    if (typeof f.status === "string") input.status = f.status.toUpperCase();
    if (typeof f.vendor === "string") input.vendor = f.vendor;
    if (typeof f.productType === "string") input.productType = f.productType;
    if (typeof f.handle === "string") input.handle = f.handle;
    return {
      query: `mutation($input: ProductInput!) { productUpdate(input: $input) { product { id title } userErrors { field message } } }`,
      variables: { input },
    };
  }
  if (op.mutation === "productVariantsBulkUpdate" && op.productId && Array.isArray(op.variants)) {
    const variants = op.variants.map((v) => {
      const o: Record<string, unknown> = { id: v.id };
      if (v.price !== undefined) o.price = String(v.price);
      if (typeof v.sku === "string") o.sku = v.sku;
      if (typeof v.title === "string") o.optionValues = [{ optionName: "Title", name: v.title }];
      return o;
    });
    return {
      query: `mutation($productId: ID!, $variants: [ProductVariantsBulkInput!]!) { productVariantsBulkUpdate(productId: $productId, variants: $variants) { product { id } productVariants { id price } userErrors { field message } } }`,
      variables: { productId: op.productId, variants },
    };
  }
  if (op.mutation === "productCreate") {
    const f = op.fields ?? {};
    const input: Record<string, unknown> = {};
    if (typeof f.title === "string") input.title = f.title;
    if (typeof f.description === "string") input.descriptionHtml = f.description;
    if (typeof f.vendor === "string") input.vendor = f.vendor;
    if (typeof f.productType === "string") input.productType = f.productType;
    if (typeof f.status === "string") input.status = f.status.toUpperCase();
    if (Array.isArray(f.tags)) input.tags = f.tags;
    return {
      query: `mutation($input: ProductInput!) { productCreate(input: $input) { product { id title } userErrors { field message } } }`,
      variables: { input },
    };
  }
  if (op.mutation === "productDelete" && op.id) {
    return {
      query: `mutation($id: ID!) { productDelete(id: $id) { deletedProductId userErrors { field message } } }`,
      variables: { id: op.id },
    };
  }
  if (op.mutation === "collectionUpdate" && op.id) {
    const f = op.fields ?? {};
    const input: Record<string, unknown> = { id: op.id };
    if (typeof f.title === "string") input.title = f.title;
    if (typeof f.description === "string") input.descriptionHtml = f.description;
    return {
      query: `mutation($input: CollectionInput!) { collectionUpdate(input: $input) { collection { id title } userErrors { field message } } }`,
      variables: { input },
    };
  }
  return null;
}

async function executeOp(op: Op): Promise<{ ok: boolean; error: string }> {
  if (op.kind === "supplier") {
    const base = env("MAKEFUN_SUPPLIER_URL");
    if (!base) return { ok: false, error: "no supplier url" };
    const path = typeof op.path === "string" && op.path.startsWith("/") ? op.path : "/";
    const method = op.method === "POST" || op.method === "PUT" || op.method === "PATCH" ? op.method : "GET";
    const r = await fetchJson(base.replace(/\/$/, "") + path, {
      method,
      headers: { "Content-Type": "application/json" },
      body: method === "GET" ? undefined : JSON.stringify(op.body ?? {}),
    }, SHOP_TIMEOUT_MS);
    return r.ok ? { ok: true, error: "" } : { ok: false, error: "supplier " + r.status + " " + r.text.slice(0, 200) };
  }
  if (op.kind !== "graphql") return { ok: false, error: "unknown op kind " + String(op.kind) };
  if (!op.mutation || !ALLOWED_MUTATIONS.has(op.mutation)) return { ok: false, error: "mutation not allowed: " + String(op.mutation) };
  const built = buildMutation(op);
  if (!built) return { ok: false, error: "bad mutation args" };
  const r = await gql(built.query, built.variables);
  if (r.errors) return { ok: false, error: truncate(JSON.stringify(r.errors), 300) };
  const d = r.data as Record<string, unknown> | null;
  if (!d) return { ok: false, error: "empty graphql response" };
  const payload = (d[op.mutation] ?? {}) as Record<string, unknown>;
  const userErrors = payload.userErrors as Array<{ message?: string }> | undefined;
  if (userErrors?.length) return { ok: false, error: userErrors.map((e) => e.message ?? "").join("; ").slice(0, 300) };
  return { ok: true, error: "" };
}

async function executePlan(ops: Op[]): Promise<Array<{ op: Op; ok: boolean; error: string }>> {
  const results: Array<{ op: Op; ok: boolean; error: string }> = [];
  for (const op of ops) {
    if (timeLeftMs() < 20_000) {
      results.push({ op, ok: false, error: "deadline" });
      continue;
    }
    const r = await executeOp(op);
    results.push({ op, ok: r.ok, error: r.error });
  }
  return results;
}

// ---------------------------------------------------------------- verification
function expectedValues(ops: Op[]): Array<{ id: string; field: string; expected: string }> {
  const out: Array<{ id: string; field: string; expected: string }> = [];
  for (const op of ops) {
    if (op.mutation === "productUpdate" && op.id && op.fields) {
      for (const [k, v] of Object.entries(op.fields)) {
        if (k === "tags") out.push({ id: op.id, field: "tags", expected: Array.isArray(v) ? v.join(",") : String(v) });
        else out.push({ id: op.id, field: k, expected: String(v) });
      }
    }
    if (op.mutation === "productVariantsBulkUpdate" && Array.isArray(op.variants)) {
      for (const v of op.variants) {
        if (v.price !== undefined) out.push({ id: String(v.id ?? ""), field: "price", expected: String(v.price) });
        if (typeof v.sku === "string") out.push({ id: String(v.id ?? ""), field: "sku", expected: v.sku });
      }
    }
  }
  return out;
}

function verify(products: Product[], expected: Array<{ id: string; field: string; expected: string }>): string[] {
  const diffs: string[] = [];
  const byId = new Map(products.map((p) => [p.id, p]));
  const variantById = new Map<string, Product["variants"][number]>();
  for (const p of products) for (const v of p.variants) variantById.set(v.id, v);
  for (const e of expected) {
    const p = byId.get(e.id);
    if (e.field === "price" || e.field === "sku") {
      const v = variantById.get(e.id);
      if (!v) { diffs.push("variant " + e.id + " not found for " + e.field); continue; }
      const actual = e.field === "price" ? v.price : v.sku;
      if (actual !== e.expected) diffs.push(e.id + " " + e.field + ": expected " + e.expected + " got " + actual);
      continue;
    }
    if (!p) { diffs.push("product " + e.id + " not found"); continue; }
    let actual = "";
    if (e.field === "title") actual = p.title;
    else if (e.field === "description" || e.field === "descriptionHtml") actual = p.description;
    else if (e.field === "status") actual = p.status;
    else if (e.field === "vendor") actual = p.vendor;
    else if (e.field === "productType") actual = p.productType;
    else if (e.field === "handle") actual = p.handle;
    else if (e.field === "tags") actual = p.tags.join(",");
    else continue;
    if (e.field === "status") {
      if (actual.toUpperCase() !== e.expected.toUpperCase()) diffs.push(e.id + " status: expected " + e.expected + " got " + actual);
    } else if (actual !== e.expected) {
      diffs.push(e.id + " " + e.field + ": expected " + truncate(e.expected, 120) + " got " + truncate(actual, 120));
    }
  }
  return diffs;
}

// ---------------------------------------------------------------- main
async function main(): Promise<void> {
  const promptFile = env("MAKEFUN_PROMPT_FILE") || "/input/prompt.txt";
  let brief = "";
  try {
    brief = fs.readFileSync(promptFile, "utf8");
  } catch {
    brief = "";
  }

  const listed = await listModels();
  let model = "";
  if (listed.models.length) {
    model = listed.models.find((m) => /claude/i.test(m)) ?? listed.models[0];
  }
  // If model listing failed, still try once with a generic name on the
  // available route so the model-call floor can be met.
  const routeBase = listed.base;
  const routeKind = listed.kind;
  if (!model && routeBase) model = routeKind === "anthropic" ? "claude-sonnet-4-5" : "gpt-4o-mini";

  const candidates = parseCandidates(brief);
  const schema = await introspectShop();
  const products = await fetchProducts();
  const extra = await fetchExtraEntities(brief);
  const snapshot = truncate(JSON.stringify(products), 8000);

  let applied = 0;
  let failed = 0;
  let verifiedOk = false;
  let lastDiffs: string[] = [];

  if (model && routeBase) {
    const reply = await callModel(routeBase, routeKind, model, PLAN_SYSTEM, planPrompt(brief, schema, snapshot, candidates, extra));
    const plan = extractJsonObject(reply) as { operations?: Op[] } | null;
    const ops = Array.isArray(plan?.operations) ? (plan!.operations as Op[]).slice(0, 30) : [];
    const results = await executePlan(ops);
    for (const r of results) {
      if (r.ok) applied++;
      else failed++;
    }
    const after = await fetchProducts();
    lastDiffs = verify(after, expectedValues(ops));
    verifiedOk = lastDiffs.length === 0;

    // One repair round only: if verification found diffs, time remains,
    // and the model-call budget allows it.
    if (lastDiffs.length > 0 && modelCalls < MAX_MODEL_CALLS && timeLeftMs() > 40_000) {
      const fixPrompt = [
        "The previous plan was executed but verification found these mismatches:",
        lastDiffs.slice(0, 20).join("\n"),
        "",
        "Current snapshot:",
        truncate(JSON.stringify(after), 6000),
        "",
        "Original brief:",
        truncate(brief, 6000),
        "",
        "Return exactly one JSON object {\"operations\": [...]} with only the repair operations needed.",
      ].join("\n");
      const reply2 = await callModel(routeBase, routeKind, model, PLAN_SYSTEM, fixPrompt);
      const plan2 = extractJsonObject(reply2) as { operations?: Op[] } | null;
      const ops2 = Array.isArray(plan2?.operations) ? (plan2!.operations as Op[]).slice(0, 30) : [];
      const results2 = await executePlan(ops2);
      for (const r of results2) {
        if (r.ok) applied++;
        else failed++;
      }
      const after2 = await fetchProducts();
      lastDiffs = verify(after2, expectedValues([...ops, ...ops2]));
      verifiedOk = lastDiffs.length === 0;
    }
  }

  const summary = {
    applied,
    failed,
    verified: verifiedOk,
    diffs: lastDiffs.slice(0, 10),
    modelCalls,
    candidates: candidates.length,
    productsSeen: products.length,
    briefBytes: brief.length,
  };
  console.log(JSON.stringify(summary));
}

main()
  .then(() => process.exit(0))
  .catch(() => {
    // Never exit non-zero: partial state is still scored.
    console.log(JSON.stringify({ applied: 0, failed: 0, verified: false, diffs: ["fatal"], modelCalls, candidates: 0, productsSeen: 0, briefBytes: 0 }));
    process.exit(0);
  });
