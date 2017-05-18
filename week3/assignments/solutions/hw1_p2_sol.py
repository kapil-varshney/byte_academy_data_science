import requests
import json
from bs4 import BeautifulSoup



class data_acquisition():
    def __init__(self):
        self.all_links = []
        self.base_url = "http://data-interview.enigmalabs.org/companies/"
        self.static_url = "http://data-interview.enigmalabs.org/companies/"
        self.json_data = {}
        self.flag = True
        
    def soupify(self,url):
        self.page = requests.get(url)
        if('200' in str(self.page)):
            BeautifulSoup(self.page.content, 'html.parser')
            self.soup = BeautifulSoup(self.page.content, 'html.parser')
            return self.soup;
        else:
            return False;
    
    def fetch_tables(self,soup):
        self.first_page_tables = soup.find('table').find('tbody').find_all('tr')
        for i in self.first_page_tables:
            self.lnk = i.a['href'].split('/')[2]
            self.all_links.append(self.lnk)
    
    def next_page(self,soup):
        self.ite = soup.find('ul',class_='pagination')
        self.nx = self.ite.find('li',class_='next').a['href']
        if(self.nx!='#'):
            self.nx = self.nx.split('/')[2]
            return self.nx
        else:
            return False
    
    
    def encode_url(self,uri):
        return(uri.replace(' ','%20'))





# Main Program
get_data = data_acquisition()

while(get_data.flag):
    soup_obj = get_data.soupify(get_data.base_url)
    if(soup_obj!=False):
        get_data.fetch_tables(soup_obj)
        nxt = get_data.next_page(soup_obj)
        if(nxt==False):
            get_data.flag = False
            break
        get_data.base_url = get_data.static_url+nxt

# Extract individual table info
print("No of links extracted {}".format(len(get_data.all_links)))
print("Data acquisition in progress...")
count = 0;
for i in get_data.all_links:
    soup_obj = get_data.soupify(get_data.static_url + get_data.encode_url(i))
    ap = soup_obj.find('table').find('tbody').find_all('tr')
    identifier = ap.pop(0)
    identifier = identifier.td.next_sibling.next_sibling.get_text()
    get_data.json_data.update({identifier:[]})
    field = {}
    for j in ap:
        field[j.td.get_text()] = j.td.next_sibling.next_sibling.get_text()
    get_data.json_data[identifier] = field    
with open('solution.json','w') as dataset:
    json.dump(get_data.json_data,dataset)

print("File write complete")
