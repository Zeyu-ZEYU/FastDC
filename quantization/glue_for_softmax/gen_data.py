import random
import time
import warnings
from random import randrange

import numpy as np
import openai
import pandas as pd
from datasets import load_dataset
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

warnings.filterwarnings("ignore")

path = "./"
input_data_filename = "signalmedia-1m.jsonl.gz"
preprocessed_data_filename = "signalmedia_news_dataset_sample.csv"
processed_data_filename = "signalmedia_news_dataset_sample_classified.csv"
output_data_json_filename = "news_classification.json"
output_data_csv_filename = "news_classification.csv"

#### OpenAI API Key ####
openai.api_key = ""

#### OpenAI model ####
model_name = "gpt-3.5-turbo"


# Reading zipped JSONL data as a Pandas DataFrame
raw_news_df = pd.read_json(f"{path}{input_data_filename}", lines=True)


# Selecting "News" records
raw_news_df2 = raw_news_df[raw_news_df["media-type"] == "News"]
# Shuffling the dataset
raw_news_df3 = raw_news_df2.sample(frac=1)
# Selecting top 1000 records/news articles
raw_news_df4 = raw_news_df3.head(1000)
# Saving the preprocessed data as a CSV file
raw_news_df4.to_csv(f"{path}{preprocessed_data_filename}", index=False)
# Loading the preprocessed data as a Pandas DataFrame
prep_news_df = pd.read_csv(f"{path}{preprocessed_data_filename}")

# Defining bot behavior and instructing
SYSTEM_PROMPT = """You are ChatGPT, an intelligent bot. I will give you a news article. You have to classify the news into one of the 43 categories."""

USER_PROMPT_1 = """Are you clear about your role?"""

ASSISTANT_PROMPT_1 = """Sure, I'm ready to help you with your news classification task. Please provide me with the necessary information to get started."""

# Few Shot Prompting
PROMPT = """
Categories:

U.S. NEWS
COMEDY
PARENTING
WORLD NEWS
CULTURE & ARTS
TECH
SPORTS
ENTERTAINMENT
POLITICS
WEIRD NEWS
ENVIRONMENT
EDUCATION
CRIME
SCIENCE
WELLNESS
BUSINESS
STYLE & BEAUTY
FOOD & DRINK
MEDIA
QUEER VOICES
HOME & LIVING
WOMEN
BLACK VOICES
TRAVEL
MONEY
RELIGION
LATINO VOICES
IMPACT
WEDDINGS
COLLEGE
PARENTS
ARTS & CULTURE
STYLE
GREEN
TASTE
HEALTHY LIVING
THE WORLDPOST
GOOD NEWS
WORLDPOST
FIFTY
ARTS
DIVORCE
ESG

If you don't know the category, response "OTHERS".

Output Format:
Category name

Examples:
1. News: New Product Gives Marketers Access to Real Keywords, Conversions and Results Along With 13 Months of Historical Data

SAN FRANCISCO, CA -- (Marketwired) -- 09/17/15 -- Jumpshot, a marketing analytics company that uses distinctive data sources to paint a complete picture of the online customer journey, today announced the launch of Jumpshot Elite, giving marketers insight into what their customers are doing the 99% of the time they're not on your site. For years, marketers have been unable to see what organic and paid search terms users were entering, much less tie those searches to purchases. Jumpshot not only injects that user search visibility back into the market, but also makes it possible to tie those keywords to conversions -- for any web site.

"Ever since search engines encrypted search results, marketers have been in the dark about keywords, impacting not only the insight into their own search investments, but also their ability to unearth high converting keywords for their competitors," said Deren Baker, CEO of Jumpshot. "Our platform eliminates the hacks, assumptions, and guesswork that marketers are doing now and provides real data: actual searches tied to actual conversions conducted by real people with nothing inferred."

Unlike other keyword research tools that receive data through the Adwords API or send bots to cobble together various data inputs and implied metrics, Jumpshot leverages its panel of over 115 million global consumers to analyze real search activity. As a result, Jumpshot is able to provide companies with actionable data to improve the ROI of their search marketing campaigns, SEO tactics and content marketing initiatives.

Available today, Jumpshot Elite provides 13 months of backward-looking data as well as:

Access to real queries used by searchers

Paid and organic results for any website

Visibility into organic keywords, eliminating the "not provided" outcome in web analytics

Real user queries, clicks and transactions instead of machine-generated clicks with inferred results

Ability to tie keywords to real transactions on any website

Variable attribution models and lookback windows

Launched in January, 2015, Jumpshot grew out of the ambitions of a group of smart marketers and data scientists who were frustrated about the limitations of the data they had access to, and excited about the opportunity to provide new insights into online behavior.

The company uses distinctive data sources to paint a complete picture of the online world for businesses, from where customers spend time online to what they do there and how they get from place to place. By tracking the online customer journey down to each click, Jumpshot reveals how and why customers arrive at purchase decisions. The company tracks more data in more detail than other services, tracking 160 billion monthly clicks generated by its extensive data panel.

About Jumpshot

Jumpshot is a marketing analytics platform that reveals the entire customer journey -- from the key sources of traffic to a site, to browsing and buying behavior on any domain. With a panel of 115 million users, Jumpshot provides marketers with the insight to understand what their customers are doing the 99% of the time they're not on their own site -- a scope of information never before attainable. Jumpshot was founded in 2015 and is headquartered in San Francisco.

For more information, please visit www.jumpshot.com.

Image Available: http://www2.marketwire.com/mw/frame_mw?attachid=2889222

Kelly Mayes

The Bulleit Group

615-200-8845

Published Sep. 17, 2015

Copyright © 2015 SYS-CON Media, Inc. — All Rights Reserved.

Syndicated stories and blog feeds, all rights reserved by the author.

Output: TECHNOLOGY

2. News: SOURCE Harwood Feffer LLP

NEW YORK

On July 21, 2015

On this news, VASCO stock nearly 33% and has not recovered.

Our investigation concerns whether the Company board of directors has breached its fiduciary duties to shareholders, grossly mismanaged the Company, and/or committed abuses of control in connection with the foregoing.

If you own VASCO shares and wish to discuss this matter with us, or have any questions concerning your rights and interests with regard to this matter, please contact:

Robert I. Harwood, Esq.

Harwood Feffer

The law firm responsible for this advertisement is Harwood Feffer LLP (www.hfesq.com). Prior results do not guarantee or predict a similar outcome with respect to any future matter.

Logo - http://photos.prnewswire.com/prnh/20120215/MM54604LOGO

To view the original version on PR Newswire, visit:http://www.prnewswire.com/news-releases/harwood-feffer-llp-announces-investigation-of-vasco-data-security-international-inc-300149371.html

©2015 PR Newswire. All Rights Reserved.

Output: BUSINESS

3. {}
Output:
"""


# Decorator for automatic retry requests
@retry(
    retry=retry_if_exception_type(
        (
            openai.error.APIError,
            openai.error.APIConnectionError,
            openai.error.RateLimitError,
            openai.error.ServiceUnavailableError,
            openai.error.Timeout,
        )
    ),
    # Function to add random exponential backoff to a request
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(10),
)

# Function to invoke Open AI's Chat Complete AI
def chat_completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


# Function to pass model name and user prompts and receive response
def openai_chat_completion_response(USER_PROMPT_2):
    response = chat_completion_with_backoff(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_1},
            {"role": "assistant", "content": ASSISTANT_PROMPT_1},
            {"role": "user", "content": USER_PROMPT_2},
        ],
    )

    return response["choices"][0]["message"]["content"].strip(" \n")


# Function to classify news articles
def predict_news_category(news_body):
    # Add news article to the prompt
    NEWS = news_body
    FINAL_PROMPT = PROMPT.format(NEWS)
    # Send prompt for inference
    try:
        classify_news = openai_chat_completion_response(FINAL_PROMPT)
    except:
        # Output "NA" if the request fails
        classify_news = "NA"
    time.sleep(20)
    return classify_news


prep_news_df2 = prep_news_df.iloc[0:100, :].copy()
prep_news_df2["predicted_category"] = prep_news_df2["content"].apply(lambda x: predict_news_category(x))
prep_news_df2.to_csv(f"{path}{processed_data_filename}", index=False)
prep_news_df2 = pd.read_csv(f"{path}{processed_data_filename}")

pred_cat_freq_dist = prep_news_df2["predicted_category"].value_counts(dropna=False).sort_values(ascending=False).reset_index()
pred_cat_freq_dist = pred_cat_freq_dist.rename(columns={"index": "predicted_category", "predicted_category": "count"})
# Merging new news categories with existing ones
prep_news_df2["predicted_category"] = np.where(
    prep_news_df2["predicted_category"] == "TECHNOLOGY", "TECH", prep_news_df2["predicted_category"]
)
prep_news_df2["predicted_category"] = np.where(
    prep_news_df2["predicted_category"] == "SPACE", "SCIENCE", prep_news_df2["predicted_category"]
)
prep_news_df2["predicted_category"] = np.where(
    prep_news_df2["predicted_category"] == "FINANCE", "MONEY", prep_news_df2["predicted_category"]
)
prep_news_df2["predicted_category"] = np.where(
    prep_news_df2["predicted_category"] == "MARKETING & ADVERTISING", "OTHERS", prep_news_df2["predicted_category"]
)
prep_news_df2["predicted_category"] = np.where(
    prep_news_df2["predicted_category"] == "ARTS & CULTURE", "CULTURE & ARTS", prep_news_df2["predicted_category"]
)
pred_cat_freq_dist = prep_news_df2["predicted_category"].value_counts(dropna=False).sort_values(ascending=False).reset_index()
pred_cat_freq_dist = pred_cat_freq_dist.rename(columns={"index": "predicted_category", "predicted_category": "count"})

# Creating instruction against each news article / news category pairs
prep_news_df2[
    "instruction"
] = """Categorize the news article into one of the 18 categories:

WORLD NEWS
COMEDY
POLITICS
TECH
SPORTS
BUSINESS
OTHERS
ENTERTAINMENT
CULTURE & ARTS
FOOD & DRINK
MEDIA
RELIGION
MONEY
HEALTHY LIVING
SCIENCE
EDUCATION
CRIME
ENVIRONMENT

"""

# Removing null news category records
prep_news_df3 = prep_news_df2[~prep_news_df2["predicted_category"].isna()]

# Renaming and selecting relevant columns
prep_news_df4 = prep_news_df3.rename(columns={"content": "input", "predicted_category": "output"})
output_news_df = prep_news_df4[["instruction", "input", "output"]]

news_json = output_news_df.to_json(orient="records", lines=True).splitlines()

with open(f"{path}{output_data_json_filename}", "w") as f:
    for line in news_json:
        f.write(f"{line}\n")

# Saving as a CSV file
output_news_df.to_csv(f"{path}{output_data_csv_filename}", index=False)
