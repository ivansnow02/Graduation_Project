import puppeteer from 'puppeteer';
const browser = await puppeteer.launch({ headless: 'new' });
const page = await browser.newPage();
await page.goto('http://localhost:5173/');
await page.waitForSelector('.sidebar');
const rects = await page.evaluate(() => {
  return {
    wrapper: document.querySelector('.layout-wrapper')?.getBoundingClientRect(),
    sidebar: document.querySelector('.sidebar')?.getBoundingClientRect(),
    header: document.querySelector('.sidebar-header')?.getBoundingClientRect(),
    content: document.querySelector('.sidebar-content')?.getBoundingClientRect(),
    footer: document.querySelector('.sidebar-footer')?.getBoundingClientRect()
  };
});
console.log(JSON.stringify(rects, null, 2));
await browser.close();
