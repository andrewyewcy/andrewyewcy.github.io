## Downloading and converting data files from Bixi systematically using Python
By: Andrew Yew

**Motivation**

Often times when collecting data for a project, we encounter consumer facing websites that do not have full Application Programming Interfaces (APIs) that allow for data to be accessed and downloaded in a programatic way.
What this usually means is that we as data scientists have to manually click the download buttons on a website and provide a directory for the file to be saved on a local drive.

For one or two files this is maneagable but as companies continue to publish more open data annually, it is not unusual to find websites such as the one shown below belonging to [Bixi](https://bixi.com/en/open-data), a bike-share company in the city of Montreal, accessed 2023-May-17.

![001](/assets/images/001.png){:class="img-responsive"}

In the image above, we see a file for each year, with the most recent year having a file for each month. Downloading the data manually would mean having to repeatedly click download 16 times, not including the time for renaming and consolidating the files later into a data folder.

In this notebook, we will explore a more systematic way to access and download all data links within a website using requests and BeautifulSoup in a Jupyter Notebook running Python.

The notebook is split into two parts:
- part 1: how to web-scrape all download links from the Bixi website
- part 2: using the web-scraped links, how to download and store the data as a Comma Separated Value (CSV) file.

## Part 1: Web-scrape all download links from Bixi website

We begin as most jupyter notebooks do, with importing required packages.


```python
import pandas as pd # for general data processing
import requests # for scraping data from websites
from bs4 import BeautifulSoup # for converting scraped data into structured html
```

Before performing any web-scraping, the first step is to identify the website(s) from which to be scraped from. For this notebook, the website where Bixi hosts its data was identified and stored in the "url" variable below.


```python
# Store target website as a variable
url = "https://www.bixi.com/en/open-data"
```

To gather data from a website, we will be using the get method within the requests package. Further documentation on requests is available [here](https://docs.python-requests.org/en/latest/index.html).


```python
# Send a GET request to gather a response
response = requests.get(url = url, allow_redirects =True)
```

Then we examine the status code of the response to determine if it was successful.


```python
print(f"Response status code: {response.status_code}, status: {response.reason}")
```

    Response status code: 200, status: OK


Then, we can use the "text" method on the response object to visually examine the contents of the response. It was expected to receive a large text blob containing all the elements of the provided website. To avoid the notebook from becoming too long, only the first 100 characters of the response were displayed below.


```python
response.text[0:100]
```




    '<!doctype html>\n<html class="no-js" lang="en" data-scrollbar>\n\t<head>\n\t\t<meta charset="utf-8">\n<meta'



As data scientists, most of the contents within response are not useful as they pertain to the design and layout of the website. How then do we identify the specific components that contain the data we are seeking to scrape?

To answer this, we may perform an element inspection on the specific desired part of a website using a web browser. Right click on the element containing the data ("Year 2021" below), then select "Inspect".   

![002](/assets/images/002.png){:class="img-responsive"}

In the element inspector that appears, the element and its corresponding code block was highlighted. In the code block, the url embedded within the element can be identified after the "a" html tag. Clicking on this url will lead to the download of a zip file containing the data. 

The other data containing urls were observed to be above and below the highlighted code block. For this case, note that all the other data urls contain the string 'amazonaws' as a common pattern. This means that the data is actually stored on an Amazon S3 bucket. To facilitate the web-scraping of all urls that contain data, the string pattern 'amazonaws' will be used to identify such urls from the response object downloaded earlier. 

Side note: although there is another Python package that specializes in dealing with Amazon S3 buckets, the method presented in this notebook is more generizable to other data stored outside of Amazon S3. 

![003](/assets/images/003.png){:class="img-responsive"}

After identifying 'amazonaws' as a string pattern that uniquely identifies data download urls among the web-scraped text blob, the next step is to use the BeautifulSoup package to turn the text blob into structured HTML(soup object) for querying.


```python
# Store the string pattern as a variable
url_string_pattern = 'amazonaws'

# Convert response text blob into structured HTML format
soup = BeautifulSoup(response.text, 'html.parser')

# Check type of converted response text
print(f"Converted response has type: {type(soup)}")
```

    Converted response has type: <class 'bs4.BeautifulSoup'>


As the soup object is a collection of structured HTML tags, the HTML tag 'a' can be used to identify all urls within the soup object.


```python
# Find all tags that are 'a' using the find_all method
url_tags = soup.find_all('a')
```

As seen below, the HTML tag 'a' along with any urls have been extracted into a list.


```python
print(f"Tag object has type: {type(url_tags[0])}")
print("")

# Visually examine first 5 tags
for index, tag in enumerate(url_tags[0:5]):
    print(f"Tag {index + 1} / {len(url_tags)}: {tag}")
```

    Tag object has type: <class 'bs4.element.Tag'>
    
    Tag 1 / 99: <a class="logo" href="/en"></a>
    Tag 2 / 99: <a class="altLang" href="https://www.bixi.com/fr/donnees-ouvertes">Français</a>
    Tag 3 / 99: <a href="https://www.bixi.com/en/network-info">Network info</a>
    Tag 4 / 99: <a href="https://www.bixi.com/en/contact-us">Contact us</a>
    Tag 5 / 99: <a class="icon-facebook social" href="https://www.facebook.com/BIXImontreal/" target="_blank"></a>


Then, the url within each tag must be extracted. This was done by using the 'get' method on each tag within the 'url_tags' list. Within the get method, the string 'href' was used to  was used to identify the urls within each tag.


```python
# Initiate a blank list to store extracted url
url_list = list()

# Loop through each tag to extract urls using get method
for tag in url_tags:
    url_list.append(tag.get('href'))
```


```python
# Visual examination of the urls extracted from tags
print(f"Extracted urls from tags:")

# Visually examine first 5 urls extracted from tags
for index, url in enumerate(url_list[0:5]):
    print(f"URL {index+1} / {len(url_list)} : {url}")
```

    Extracted urls from tags:
    URL 1 / 99 : /en
    URL 2 / 99 : https://www.bixi.com/fr/donnees-ouvertes
    URL 3 / 99 : https://www.bixi.com/en/network-info
    URL 4 / 99 : https://www.bixi.com/en/contact-us
    URL 5 / 99 : https://www.facebook.com/BIXImontreal/


Finally, the last step is to filter the list for only the urls that lead to data download, which in this case are urls that contain the string pattern 'amazonaws'.


```python
# To use pandas str.contains() method, the list of extracted urls was first converted into a DataFrame
url_df = pd.DataFrame(url_list, columns = ['extracted_url'])
```


```python
# Examine the extracted urls
display(url_df.head())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>extracted_url</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>/en</td>
    </tr>
    <tr>
      <th>1</th>
      <td>https://www.bixi.com/fr/donnees-ouvertes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>https://www.bixi.com/en/network-info</td>
    </tr>
    <tr>
      <th>3</th>
      <td>https://www.bixi.com/en/contact-us</td>
    </tr>
    <tr>
      <th>4</th>
      <td>https://www.facebook.com/BIXImontreal/</td>
    </tr>
  </tbody>
</table>
</div>


After converting the list of urls into a DataFrame, pandas was used to remove any null values and keep only urls that contain the string pattern 'amazonaws'.


```python
# Drop any null values since null values would mean no urls
url_df.dropna(inplace = True)

# Define filter condition to keep only urls that contain the string pattern
# Apply lower case first before performing a string match
cond1 = url_df['extracted_url'].str.lower().str.contains(url_string_pattern)

# Use the defined condition to filter the extracted url list
url_df = url_df.loc[cond1].reset_index(drop = True).copy()
```

To summarize, starting from the Bixi website, we have identified and gathered all 16 data download urls within the BIXI website without having to visually identify and click on each link within the website. In part 2, we will explore how to download the data contained within each of the 16 urls, unzip the data files, then combine the data into a single CSV file.


```python
print(f"The number of data download urls extracted is {url_df.shape[0]}, which is 100% of all data download urls visible on the Bixi website.")

# Visually examine first 5 data download urls
print("Visually examine first 5 filtered urls")
for index, url in enumerate(url_df['extracted_url'].to_list()[0:5]):
    print(f"URL {index+1} / {len(url_list)} : {url}")
```

    The number of data download urls extracted is 16, which is 100% of all data download urls visible on the Bixi website.
    Visually examine first 5 filtered urls
    URL 1 / 99 : https://sitewebbixi.s3.amazonaws.com/uploads/docs/biximontrealrentals2014-f040e0.zip
    URL 2 / 99 : https://sitewebbixi.s3.amazonaws.com/uploads/docs/biximontrealrentals2015-69fdf0.zip
    URL 3 / 99 : https://sitewebbixi.s3.amazonaws.com/uploads/docs/biximontrealrentals2016-912f00.zip
    URL 4 / 99 : https://sitewebbixi.s3.amazonaws.com/uploads/docs/biximontrealrentals2017-d4d086.zip
    URL 5 / 99 : https://sitewebbixi.s3.amazonaws.com/uploads/docs/biximontrealrentals2018-96034e.zip


## Part 2: Downloading the files from each link
