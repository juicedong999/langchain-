from playwright.sync_api import sync_playwright
import time

print("开始启动 Playwright...")
with sync_playwright() as p:
    try:
        # 强制弹出实体浏览器窗口
        print("正在启动 Chromium 浏览器...")
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        
        print("正在访问百度...")
        page.goto("https://www.baidu.com", timeout=30000)
        
        print(f"✅ 访问成功！页面标题是: {page.title()}")
        time.sleep(3) # 停顿3秒让你看清画面
        
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        
    finally:
        if 'browser' in locals():
            browser.close()
        print("测试结束。")