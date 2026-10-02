# References

Style: **IEEE**, numbered in order of first use in `docs/review_preprocessing_section.md`.

**Verification status.** This sandbox blocks Figshare, Nature, PubMed Central, doi.org and Crossref
(HTTP 403 from the egress proxy), so no entry could be checked against its primary record. Every
entry is therefore marked **[verify]**. The note after each entry says where its details came from:

- "web search": the title, journal and DOI appeared in search-engine results, but the page itself
  could not be opened.
- "project brief": the DOI was given in the project instructions and matches the dataset files we
  use, but the Figshare record could not be opened.
- "from memory": a standard reference whose details were not checked here. Check these most
  carefully.

Please confirm each one (open the DOI) before submission and then remove the tag.

## Data

[1] S. K. Kuruva *et al.*, "Quality controlled, reliable groundwater level data with corresponding
specific yield over India," Figshare, dataset, version 3, CC BY 4.0. doi:10.6084/m9.figshare.29293877.v3.
**[verify]** *Project brief for the DOI. The author list is taken from the associated paper [2]
(web search); confirm the authors, year and exact title on the Figshare record. The dataset's
ReadMe.docx gives no citation text.*

[2] S. K. Kuruva *et al.*, "Quality controlled, reliable groundwater level data with corresponding
specific yield over India," *Scientific Data*, 2025. doi:10.1038/s41597-025-05899-5. **[verify]**
*Web search (Nature and PubMed 41034238 listings). First author and affiliation (Interdisciplinary
Centre for Water Research, Indian Institute of Science, Bengaluru) are from the search snippet.
Confirm the full author list, volume and article number.*

[3] C. Funk *et al.*, "The climate hazards infrared precipitation with stations—a new environmental
record for monitoring extremes," *Scientific Data*, vol. 2, Art. no. 150066, 2015.
doi:10.1038/sdata.2015.66. **[verify]** *DOI given in the project brief; other details from memory.*

[4] C. Funk, P. Peterson, L. Harrison *et al.*, "The Climate Hazards Center Infrared Precipitation
with Stations, Version 3," *Scientific Data*, vol. 13, Art. no. 718, 2026.
doi:10.1038/s41597-026-07096-4. **[verify]** *Web search (Nature, PMC and USGS listings).*

[5] Climate Hazards Center, "Climate Hazards Center Infrared Precipitation with Stations version 3
(CHIRPS3) data repository," University of California, Santa Barbara, 2025. doi:10.15780/G2JQ0P.
Files used: https://data.chc.ucsb.edu/products/CHIRPS/v3.0/monthly/global/tifs/ (accessed
Oct. 2026). **[verify]** *The repository DOI is from web search. The file URL was accessed directly,
and the README there (`README-CHIRPSv3.0.txt`) names C. Funk as the contact but gives no citation
string.*

## Methods

[6] H. Theil, "A rank-invariant method of linear and polynomial regression analysis, I, II, III,"
*Proc. Koninklijke Nederlandse Akademie van Wetenschappen*, vol. 53, pp. 386–392, 521–525,
1397–1412, 1950. **[verify]** *From memory.*

[7] P. K. Sen, "Estimates of the regression coefficient based on Kendall's tau," *J. American
Statistical Association*, vol. 63, no. 324, pp. 1379–1389, 1968. doi:10.1080/01621459.1968.10480934.
**[verify]** *From memory.*

[8] Student, "The probable error of a mean," *Biometrika*, vol. 6, no. 1, pp. 1–25, 1908.
doi:10.1093/biomet/6.1.1. **[verify]** *From memory. Cited for the t-distribution behind the
small-sample adjustment (`anomaly_tadj`), which uses the standard prediction interval for a new
observation, (x − x̄) / (s·√(1 + 1/n)) ~ t with n − 1 degrees of freedom.*

[9] B. Iglewicz and D. C. Hoaglin, *How to Detect and Handle Outliers* (ASQC Basic References in
Quality Control, vol. 16). Milwaukee, WI, USA: ASQC Quality Press, 1993. **[verify]** *From memory.
Cited for the modified z-score cut-off of 3.5 used by `flag_extreme`.*

## Software

[10] W. McKinney, "Data structures for statistical computing in Python," in *Proc. 9th Python in
Science Conf. (SciPy 2010)*, 2010, pp. 56–61. doi:10.25080/Majora-92bf1922-00a. **[verify]** *From
memory. pandas 3.0.6 was used.*

[11] C. R. Harris *et al.*, "Array programming with NumPy," *Nature*, vol. 585, pp. 357–362, 2020.
doi:10.1038/s41586-020-2649-2. **[verify]** *From memory. NumPy 2.4.6 was used.*

[12] P. Virtanen *et al.*, "SciPy 1.0: Fundamental algorithms for scientific computing in Python,"
*Nature Methods*, vol. 17, pp. 261–272, 2020. doi:10.1038/s41592-019-0686-2. **[verify]** *From
memory. SciPy 1.17.1 was used.*

[13] S. Gillies *et al.*, "Rasterio: geospatial raster I/O for Python programmers," software,
https://github.com/rasterio/rasterio. **[verify]** *From memory. No DOI is cited, so the version
DOI should be added from the project's citation file. Rasterio 1.4.4 was used.*

[14] The Apache Software Foundation, "Apache Parquet" and "Apache Arrow," https://parquet.apache.org
and https://arrow.apache.org. **[verify]** *URLs from memory. pyarrow 25.0.1 was used.*

[15] J. D. Hunter, "Matplotlib: A 2D graphics environment," *Computing in Science & Engineering*,
vol. 9, no. 3, pp. 90–95, 2007. doi:10.1109/MCSE.2007.55. **[verify]** *From memory. Matplotlib
3.10.9 was used for the figures.*
