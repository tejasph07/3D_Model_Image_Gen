// Node driver: serves the stage page + model library, runs jobs in headless Chromium, writes PNGs.
// usage: node render.mjs jobs.json   (jobs.json: [{out, width, height, camera, actors, environment}, ...])
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { chromium } from "playwright-core";

const here = path.dirname(new URL(import.meta.url).pathname);
const assets = process.env.ANIMESCENE_ASSETS || "/home/user/assets";
const TYPES = { ".js": "text/javascript", ".mjs": "text/javascript", ".html": "text/html", ".json": "application/json" };
const server = http.createServer((req, res) => {
  const u = decodeURIComponent(req.url.split("?")[0]);
  const file = u.startsWith("/assets/") ? path.join(assets, u.slice(8)) : path.join(here, u === "/" ? "index.html" : u);
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404); return res.end(); }
    res.writeHead(200, { "Content-Type": TYPES[path.extname(file)] || "application/octet-stream" });
    res.end(data);
  });
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const port = server.address().port;
const exe = process.env.CHROMIUM || "/opt/pw-browsers/chromium_headless_shell-1194/chrome-linux/headless_shell";
const browser = await chromium.launch({ executablePath: exe, args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] });
const page = await browser.newPage();
page.on("console", (m) => { if (m.type() === "error" || m.type() === "warning") console.error("[page]", m.text()); });
page.on("pageerror", (e) => console.error("[page error]", e.message));
await page.goto(`http://127.0.0.1:${port}/`);
await page.waitForFunction(() => window.ready === true, null, { timeout: 60000 });
const jobs = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
for (const job of jobs) {
  const t0 = Date.now();
  if (job.inspect) {
    const out = await page.evaluate((u) => window.inspectModel(u), job.inspect);
    fs.writeFileSync(job.out, JSON.stringify(out));
  } else {
    const res = await page.evaluate((j) => window.renderJob(j), job);
    fs.writeFileSync(job.out, Buffer.from(res.beauty.split(",")[1], "base64"));
    if (res.bones) fs.writeFileSync(job.out.replace(/\.png$/, "_bones.json"), JSON.stringify(res.bones));
    for (const k of ["depth", "ids"]) {
      if (res[k]) fs.writeFileSync(job.out.replace(/\.png$/, `_${k}.png`), Buffer.from(res[k].split(",")[1], "base64"));
    }
  }
  console.log(`[render3d] ${job.out} ${Date.now() - t0}ms`);
}
await browser.close();
server.close();
