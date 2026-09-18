CREATE database GDELTTrends;
USE GDELTTrends;

-- Creating Table for GDELT GKG
CREATE TABLE GDELT_GKG (
	Week_Start DATE NOT NULL,
    Country_Code VARCHAR(10) NOT NULL,
    Category VARCHAR(100),
    Media_Volume INT,
    tone_net_sentiment FLOAT,
    tone_positive_score FLOAT,
    tone_negative_score FLOAT,
    tone_polarity FLOAT,
    tone_activity_density FLOAT,
    tone_self_group_density FLOAT,
    Source_Diversity INT,
    PRIMARY KEY (Week_Start, Country_Code, Category)
)

-- Creating table for Google Trends data
CREATE TABLE google_trends (
    Week_Start DATE NOT NULL,
    Country_Code VARCHAR(10) NOT NULL,
    Country_Name VARCHAR(100),
    Category VARCHAR(100),
    Search_Interest INT,
    PRIMARY KEY (Week_Start, Country_Code, Category)
);

-- IMPORTING DATA INTO BOTH TABLES USING GUI

SELECT COUNT(*) FROM GDELT_GKG;
SELECT COUNT(*) FROM google_trends;

