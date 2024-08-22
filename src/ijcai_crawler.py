import re
import requests
from bs4 import BeautifulSoup
from time import sleep
from requests.adapters import HTTPAdapter
import pandas as pd





ijcai_proc_url = "https://www.ijcai.org/Proceedings/2024/"

response = requests.get(ijcai_proc_url)
response.encoding = response.apparent_encoding

# parse the html
soup = BeautifulSoup(response.text, "lxml")

# select all the `div` tags of the class `paper_wrapper`
paper_html_list = soup.css.select(".paper_wrapper")

# print("Number of papers: ", len(paper_html_list))
# print("Example paper: ", paper_html_list[1])
# print("The last paper: ", paper_html_list[-1])

ijcai_paper_base_url = "https://www.ijcai.org"

# extract the paper title, authors, and url
paper_list = []
cnt = 0

s = requests.Session()
a = HTTPAdapter(max_retries=6)
s.mount('https://', a)

for paper_html in paper_html_list:
    paper = {}
    paper["paper_title"] = paper_html.css.select(".title")[0].text.strip()
    print("Paper title: ", paper["paper_title"])
    relative_url = paper_html.css.select(".details a")[1]["href"]
    full_paper_url = ijcai_paper_base_url + relative_url
    paper_details_response = s.get(url=full_paper_url)
    paper_details_response.encoding = paper_details_response.apparent_encoding
    paper_details_soup = BeautifulSoup(paper_details_response.text, "lxml")
    paper["abstract"] = paper_details_soup.css.select(".col-md-12")[0].text.strip()
    paper["abstract"] = re.sub(r"\s+", " ", paper["abstract"])
    paper_list.append(paper)
    cnt += 1
    print("Paper ", cnt, " done.")
    # print( "Sleeping for 2 seconds... ")
    # sleep(2)

print("Example paper: ", paper_list[0])

# convert to pandas dataframe
paper_df = pd.DataFrame(paper_list)

# save to the file
paper_df.to_csv("data/IJCAI_2024.csv", index=False)

