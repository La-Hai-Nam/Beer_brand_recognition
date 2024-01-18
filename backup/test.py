import requests
from bs4 import BeautifulSoup

# Make a request to the website
response = requests.get('https://www.kaufda.de/Angebote/Sternburg')

# Parse the HTML content
soup = BeautifulSoup(response.content, 'html.parser')

# Find a specific element on the page
name = soup.find('div', attrs={'class': 'offer-retailer ellipsis'})
price = soup.find('p', attrs={'class': 'offer-price ellipsis'})
# Extract the desired information from the element
nameInfo = name.text
priceInfo = price.text

print("Supermarkt:", nameInfo, "\nPrice:", priceInfo)
