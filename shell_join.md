I'd like to thank Vidwath Hebse, a former teammate, for identifying the solution to this problem.

## Introduction

We often perform joins using SQL. Consider a query - "retrieve all the medical claims records of patients who suffered from knee issues". This query can be logically divided into two queries:

1. Identify and select all the patients (IDs) who suffered from knee issues
2. Pull all the medical claims records of the selected patients based on their IDs

If the total number of records is n and the total number of distinct patients if m, the brute force solution will be of the order O(m*n). Can this be done more efficiently?

## The data

The assumption here is that we have a small subset of unique key entries or a table with distinct foreign key entries that should be retrieved from the main table. Let us assume that all the records are in the form of a text file. As a practical example, we will free data generated using [Synthea](https://synthetichealth.github.io/synthea/) for performing the join. Here are the required data sets:

- [FHIR R4](https://synthetichealth.github.io/synthea-sample-data/downloads/synthea_sample_data_fhir_r4_sep2019.zip)

## Efficient join using shell script