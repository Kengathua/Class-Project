import streamlit as st

import os
import datetime
from datetime import date
import requests
from bs4 import BeautifulSoup

import datetime
from datetime import date
import requests
from bs4 import BeautifulSoup

import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException

from pandas_datareader._utils import RemoteDataError

import csv

options = Options()
options.add_argument('--ignore-certificate-errors')
options.add_argument('--incognito')
options.add_argument('--headless')

driver = webdriver.Chrome(
    "/usr/bin/chromedriver", options=options)