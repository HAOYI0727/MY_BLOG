/* This is a script to compile inline Stylus */

const fs = require('fs');
const path = require('path');
const stylus = require('stylus');

// Recursively list files under a directory
function walk(dir) {
  const res = [];
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const e of entries) {
    const full = path.join(dir, e.name);
    if (e.isDirectory()) res.push(...walk(full));
    else res.push(full);
  }
  return res;
}

const srcDir = path.resolve('src');
if (!fs.existsSync(srcDir)) {
  console.error('No src directory found.');
  process.exit(1);
}

const files = walk(srcDir);

const styleBlockRe = /<style[^>]*lang=["']stylus["'][^>]*>([\s\S]*?)<\/style>/ig;
let failed = false;

function compileStylusContent(content, filename) {
  try {
    stylus(content).set('filename', filename).render();
    return { ok: true };
  } catch (err) {
    const msg = err.message;
    return { ok: false, error: msg };
  }
}

// Check .styl files
const stylFiles = files.filter(f => f.endsWith('.styl'));
for (const f of stylFiles) {
  const content = fs.readFileSync(f, 'utf8');
  const r = compileStylusContent(content, f);
  if (r.ok) console.log(`${f}: OK`);
  else {
    failed = true;
    console.error(`${f}: ERROR\n${r.error}`);
  }
}

// Check inline <style lang="stylus"> blocks in component files
const candidateExt = ['.astro', '.svelte', '.vue', '.html'];
for (const f of files) {
  if (!candidateExt.includes(path.extname(f))) continue;
  const text = fs.readFileSync(f, 'utf8');
  let m;
  styleBlockRe.lastIndex = 0;
  let idx = 0;
  while ((m = styleBlockRe.exec(text)) !== null) {
    idx += 1;
    const content = m[1].trim();
    const r = compileStylusContent(content, f);
    if (r.ok) console.log(`${f} [style #${idx}]: OK`);
    else {
      failed = true;
      console.error(`${f} [style #${idx}]: ERROR\n${r.error}`);
    }
  }
}

if (failed) {
  console.error('\nStylus check failed. Fix above errors.');
  process.exit(1);
} else {
  console.log('\nAll Stylus checks passed.');
  process.exit(0);
}
