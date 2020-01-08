import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import astropy.units as u

"""SpectralQuery is designed to query the Cologne Database for Molecular Spectroscopy (CDMS)
and Jet Propulsion Laboratory (JPL) for spectral line emission data."""

class SpectralQuery:

    # Initialize the class instance
    def __init__(self, database):

        # Request the database
        if database == "CDMS":
            self.DB_URL = "https://cdms.astro.uni-koeln.de/classic/entries/"
            self.DB_LINES_URL = "https://cdms.astro.uni-koeln.de"
        elif database == "JPL":
            self.DB_URL  = "https://spec.jpl.nasa.gov/ftp/pub/catalog/catdir.html"
            self.DB_LINES_URL = "https://spec.jpl.nasa.gov/ftp/pub/catalog/c"
        self.database = database
        self.page = requests.get(self.DB_URL)
        self.soup = BeautifulSoup(self.page.content, 'html.parser')

    # Method to retrieve spectral line data
    def getSpectralLines(self, molecule_tag):
        if self.database == "CDMS":
            self.link = self.soup.find("table").find(lambda tag:tag.name=="td" and molecule_tag in tag.text).find(lambda tag:tag.name == "a" and "HTML" in tag.text)['href']
            self.lines_page = requests.get(self.DB_LINES_URL + self.link)
            self.lines_soup = BeautifulSoup(self.lines_page.content, 'html.parser')
            self.link2 = self.lines_soup.find("a")['href']
            self.lines_page2 = requests.get(self.DB_LINES_URL + self.link2)
            self.lines_soup2 = BeautifulSoup(self.lines_page2.content, 'html.parser')
            self.lines = str(self.lines_soup2.find("pre"))
        elif self.database == "JPL":
            self.lines = str(BeautifulSoup(requests.get(self.DB_LINES_URL + molecule_tag + ".cat").content, 'html.parser'))

        # Clean up the line data
        # Step 1: Remove the <pre> and </pre> and any line breaks before/after them
        if "<pre>\n " in self.lines: self.lines = self.lines.replace("<pre>\n ", "")
        if "<pre>\n" in self.lines: self.lines = self.lines.replace("<pre>\n", "")
        if "\n\n</pre>" in self.lines: self.lines = self.lines.replace("\n\n</pre>", "")
        if "\n</pre>" in self.lines: self.lines = self.lines.replace("\n</pre>", "")
        if "</pre>" in self.lines: self.lines = self.lines.replace("</pre>", "")
        # Step 2: Split the lines at every line break
        self.lines = pd.Series(self.lines.split("\n"))
        self.lines = self.lines[:-1]  # Drop the last entry
        # Step 3: Separate entries that bleed toghether
        self.lines = self.lines.str.replace(r"([.]\d{4})", r"\1 ")
        # Step 4: Remove extra white space between entries
        self.lines = self.lines.str.replace("\s\s*", " ")
        # Step 4: Split at each column
        self.lines = self.lines.str.split()
        # Step 5: Create a DataFrame from the first 3 columns
        self.lines = pd.DataFrame(self.lines)
        self.lines = pd.DataFrame(self.lines[0].values.tolist())
        self.lines = pd.DataFrame(self.lines[[0, 1, 2]])
        self.lines.columns = ["frequency", "uncertainty", "intensity"]
        # It is important to note: "intensity" is given in base 10 logarithm of the integrated intensity at 300 K (in nm^2 * MHz)
