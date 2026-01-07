from playwright.sync_api import sync_playwright

def verify_trading_sim():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Load the HTML file directly
        import os
        cwd = os.getcwd()
        page.goto(f"file://{cwd}/Trading_Simulator.html")

        # Check for start screen
        page.locator("#username").fill("Tester")
        page.get_by_text("Start Simulation").click()

        # Wait for simulation to load
        page.wait_for_selector("#simulation-screen.active")

        # Verify Zoom buttons exist
        page.wait_for_selector("#btn-zoom-in-both")
        page.wait_for_selector("#btn-zoom-out-both")

        # Click Zoom + multiple times
        for _ in range(3):
            page.locator("#btn-zoom-in-both").click()
            page.wait_for_timeout(100) # wait a bit for render

        # Open Stats Modal
        page.get_by_text("Stats").click()
        page.wait_for_selector("#stats-modal")

        # Verify Export PDF button in Modal
        page.wait_for_selector("#btn-export-pdf-modal")

        # Take screenshot of Modal
        page.screenshot(path="verification/stats_modal.png")

        # Close Modal
        page.locator("#btn-close-stats").click()

        # End Simulation (Cheat by calling endSimulation in console? or just wait)
        # Actually let's just inspect the Report screen by manipulating DOM or forcing state
        page.evaluate("endSimulation()")

        page.wait_for_selector("#report-screen.active")

        # Verify Export PDF button in Report
        page.wait_for_selector("#btn-export-pdf-report")

        # Take screenshot of Report
        page.screenshot(path="verification/report_screen.png")

        browser.close()

if __name__ == "__main__":
    verify_trading_sim()
