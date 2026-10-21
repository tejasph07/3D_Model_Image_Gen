// Drive anifun.ai's Photo-to-Anime page like a visitor would: upload, pick a style, generate, wait, save.
// Uses the free guest allowance (30 credits/day, 9 per image); prints the remaining balance.
// node render3d/anifun.mjs IN.png OUT.png [style]
import { chromium } from "playwright-core";
import fs from "fs";
const [img, out, style = "Anime"] = process.argv.slice(2);
const exe = process.env.CHROME || "/opt/pw-browsers/chromium-1194/chrome-linux/chrome";
const browser = await chromium.launch({ executablePath: exe, headless: false, args: ["--headless=new", "--ignore-certificate-errors"],
  ...(process.env.HTTPS_PROXY ? { proxy: { server: process.env.HTTPS_PROXY } } : {}) });
const page = await browser.newPage({ viewport: { width: 1280, height: 1400 } });
let result = null, balance = null, error = null, uploaded = false, task = null;
page.on("response", async (r) => {
  const u = r.url();
  if (!u.includes("api.anifun.ai")) return;
  let j = null; try { j = await r.json(); } catch { return; }
  if (u.includes("check-status") && j?.data?.status === 2) result = j.data.result_image;
  if (u.includes("check-status") && j?.data?.status < 0) error = JSON.stringify(j.data);
  if (u.includes("credit/balance")) balance = j?.data?.balance;
  if (u.includes("upload-img") && j?.code === 200) uploaded = true;
  if (u.includes("generation/tasks") && j?.code === 200) task = j.result?.task_id;
  if (u.includes("generation/tasks") && j?.code !== 200) error = JSON.stringify(j);
});
await page.goto("https://anifun.ai/photo-to-anime/", { waitUntil: "domcontentloaded", timeout: 90000 });
await page.waitForSelector("input[type=file]", { state: "attached", timeout: 60000 });
await page.waitForTimeout(3000);
if (balance !== null && balance < 9) { console.log(JSON.stringify({ ok: false, error: "daily guest credits used up", balance })); process.exit(2); }
await page.waitForLoadState("load", { timeout: 90000 }).catch(() => {});
await page.waitForTimeout(5000);
for (let a = 0; a < 4 && !uploaded; a++) {  // the page may still be hydrating: give the file again until it uploads
  await page.setInputFiles("input[type=file]", img);
  for (let i = 0; i < 20 && !uploaded; i++) await page.waitForTimeout(1000);
}
await page.waitForTimeout(1500);
await page.click(`button[id="style-${style}"]`);
for (let i = 0; i < 5 && !task && !error; i++) {  // the button is live once the upload preview is in
  await page.click("button.yi-submit-btn");
  for (let k = 0; k < 10 && !task && !error; k++) await page.waitForTimeout(1000);
}
for (let i = 0; i < 300 && task && !result && !error; i++) await page.waitForTimeout(2000);
if (result) {
  const res = await page.request.get("https://temp.anifun.ai/" + result);
  fs.writeFileSync(out, await res.body());
}
console.log(JSON.stringify({ ok: !!result, uploaded, task, result, balance, error }));
await browser.close();
process.exit(result ? 0 : 1);
