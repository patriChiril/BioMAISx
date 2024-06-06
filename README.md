# BioMAISx

This repository releases the BioMAISx (*Bio*technology: *M*edia, *A*griculture, *I*nvestment, (and) *S*entiment E*x*cerpts) dataset annotated for Aspect-Based Sentiment Analysis (ABSA). It includes all code required for collecting and processing the raw data used for annotation, details on how the data was annotated, and code for post-processing the annotated data. 

## Collecting Data

The raw articles from which the quotes used in this corpus were sourced came from Factiva. You need to gain access to articles from Factiva (for a fee) and attain a user key and CID. Then to download the articles, set your key and CID to environment variables named `FACTIVA_USER_KEY` and `FACTIVA_CID`, respectively. Then you should be able to successfully run `python scripts/download-source.py`

## Preprocessing Data

From the raw text data, we filtered to articles with specific keyterms, extracted quotations from those articles, and then filtered those quotations to those within contianing terms from the desired lexicon. 

From this the quotes were reformatted for annotation with LabelStudio and proposed entities (noun chunks) were extracted using SpaCy. The code for this transformation is in scripts/preprocess-source.py

## Annotating

Relevant information and code for annotation is included in annotation/README.md