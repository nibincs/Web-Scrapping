from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
import os

options = webdriver.ChromeOptions()
options.add_argument("--headless")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
driver.get("https://coinmarketcap.com/")
time.sleep(5)
rows = driver.find_elements(By.XPATH, '//table//tbody/tr')

data = []
rank = 1

for row in rows:
    try:
        name = row.find_element(By.XPATH, './/td[3]//p').text
        if "Index" in name:
            continue
        price = row.find_element(By.XPATH, './/td[4]').text
        changehr = row.find_element(By.XPATH, './/td[5]').text
        market = row.find_element(By.XPATH, './/td[8]').text
        data.append([rank, name, price, changehr, market])
        rank += 1
        if rank > 10:
            break

    except:
        continue

driver.quit()

df = pd.DataFrame(data, columns=["Rank", "Coin Name", "Price", "24h Change", "Market Cap"])

file_name = "crypto_prices.csv"
if os.path.exists(file_name):
    try:
        os.remove(file_name)   
    except PermissionError:
        print("Close the CSV file and run again!")
        exit()

df.to_csv(file_name, index=False)
print("CSV file updated successfully!")