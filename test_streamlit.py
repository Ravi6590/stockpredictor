from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# Chrome WebDriver ka path agar required ho toh specify kar sakte ho
driver = webdriver.Chrome()

# Streamlit app open karo (ensure app running on this port)
driver.get("http://localhost:8501")

# Wait for Streamlit page to load
time.sleep(5)

# Select stock symbol from dropdown
stock_dropdown = WebDriverWait(driver, 10).until(
    EC.element_to_be_clickable((By.XPATH, "//div[contains(@class, 'stSelectbox')]//select"))
)
stock_dropdown.send_keys("AAPL")  # Apple stock select karo

# Start date select karo
start_date_input = driver.find_element(By.XPATH, "//input[@type='date']")
start_date_input.clear()
start_date_input.send_keys("2015-01-01")

# End date select karo
end_date_input = driver.find_elements(By.XPATH, "//input[@type='date']")[1]  # Dusra date input
end_date_input.clear()
end_date_input.send_keys("2023-12-31")

# Wait for predictions to be generated
time.sleep(10)

# Screenshot le lo (Proof ke liye)
driver.save_screenshot("stock_prediction_test.png")

# Test complete, close browser
driver.quit()

print("✅ Test Completed! Screenshot saved as stock_prediction_test.png")
