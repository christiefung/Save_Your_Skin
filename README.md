# Save_Your_Skin

Save Your Skin is a web application that takes a user uploaded image to return different classification of skin types (normal, dry, or oily) and recommend suitable skin care products.


Folders:
- Notebook:
  - Contains data scrapping notebooks of various websites for data collection (Image_scraper.ipynb, Skin_image_scraper.ipynb, main_amazon_scraper_3.1.ipynb
  - main_donwload_image.ipynb for downloading images from scraped Amazon image links
  - model_final.ipyng for model training
  - main_recommender.ipynb for building the recommender system
  
- images:
  - Preprocessed images

- reviews:
  - Scraped Amazon reviews with ratings and image links
  
- uploads:
  - test image
  
*Model does not perform well for pictures with a lot of noises (e.g., redundant background)
*To run on Streamlit, run product.py
