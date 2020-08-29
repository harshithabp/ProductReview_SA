from selenium import webdriver
import os
chrome_options = webdriver.chromeOptions()
chrome_options.binary_location = os.environ.get("GOOGLE_CHROME_BIN")
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-dev-shm-ussage")
chrome_options.add_argument("--no-sandbox")
driver = webdriver.Chrome(executable_path=os.environ.get("CHROMEDRIVER PATH"), chrome_options=chrome_options) 

driver.get("https://www.amazon.in/JBL-C100SI-Ear-Headphones-Black/product-reviews/B01DEWVZ2C/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews")
print(driver.page_source)