import streamlit as st
from PIL import Image
from tqdm import tqdm_notebook as tqdm
import numpy as np
import glob
from compare_similarity import compare_similarity
from keras.preprocessing.image import load_img
from img_classificattion import predict
import pandas as pd
from keras.preprocessing.image import img_to_array
import base64
st.title("Save Your Skin")
st.header("Skin type classification and Product Recommendation")
st.text("Upload a face image")


#@st.cache
#def load_data(data_path):
#    data = pd.read_csv(data_path)
#    return data
#img_path = ('/Users/ChristieFung/Desktop/Insight/skin_care_scraper-2/data2/df_new_rec.csv')

imgs = {}
classes=[]
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    result = predict(image, 'd_model.h5')
    label = np.argmax(result)
    classes=[]
    if label == 0:
        st.write("Label: Dry")
        classes += ['dry']
    elif label == 1:
        st.write("Label: Normal")
        classes += ['normal']
    elif label == 2:
        st.write("Label: Oily")
        classes += ['oily']

#    df_label = pd.read_csv('/Users/ChristieFung/Desktop/Insight/skin_care_scraper-2/data2/label.csv')
#    df_label = pd.DataFrame(classes, columns=['label'])
#    df_label.to_csv('/Users/ChristieFung/Desktop/Insight/skin_care_scraper-2/data2/label.csv')
#    df_label = pd.read_csv('/Users/ChristieFung/Desktop/Insight/skin_care_scraper-2/data2/label.csv')

    # Compare the similarity between the test images and the datase images, and then output the 3 products with the highest score
    data_dir = '../reviews'
    fs = ['11-06-2020_12-36-51_amazon_review_with_image_normal_imgfiles.csv',
     '11-06-2020_12-15-47_amazon_review_with_image_dry_imgfiles.csv',
      '11-06-2020_11-48-03_amazon_review_with_image_oily_imgfiles.csv']

# load into dataframes and concat them
    df_fs = []
    for f in fs:
        df_f = pd.read_csv('%s/%s' %(data_dir, f))
        df_fs.append(df_f)
    df_fs = pd.concat(df_fs)
    df_fs = df_fs.drop(columns=['Unnamed: 0'])

# rating parse the score
    df_fs['rating_numerical'] = [float(s.split()[0]) for s in df_fs['rating'].tolist()]
# remove duplicate in case there are
    df_fs = df_fs.drop_duplicates(subset=['img_files'])

    df_new = []

# 'img_files' columns contains string with multiple image files
    for i in tqdm(range(0, df_fs.shape[0])):
        df_f = df_fs.iloc[i]
        imgs = df_f['img_files'].split(',')
        for img in imgs:
            tmp = df_f.copy(deep=True)
            tmp['img_file'] = img
            df_new.append(tmp)

# concat
    df_new = pd.DataFrame(df_new)
    df_new = df_new.drop(columns=['img_files'])
    df_new = df_new.reset_index(drop=True)

    img_fs = glob.glob('../images/*.jpg')

# identify which images are actually from the amazon product list
# E.g., greyd0004_dry_B00OW9OU2O_pid673_1.jpg --> I only care this part 'dry_B00OW9OU2O_pid673_1'
    imgs_amazon = []
    for img_f in img_fs:
    # Get the part I care
        name = img_f.split('/')[-1]
        name = name.split('_')[1:]

    # there should be four components for the part that we like. E.g.,
    # ['oily', 'B00BEUAZTG', 'pid1022', '0.jpg']. If not, do not include them in the list
        if len(name) == 4:
            name = "_".join(name).replace('.jpg', "")
            imgs_amazon.append(name)

# remove duplicates
    imgs_amazon = list(set(imgs_amazon))

# Identify the image files that appear both in the product info as well as the image list
    imgs_amazon_comm = list(set(imgs_amazon) & set(df_new['img_file'].tolist()))

# Only get the product info that also has the images
    df_new_comm = df_new[df_new['img_file'].isin(imgs_amazon_comm)]
    df_new_comm = df_new_comm.set_index('img_file')

###loab product library
# get all available images full path first
#    img_fs = glob.glob('/Users/ChristieFung/Desktop/Insight/skin_care_scraper-2/data2/recommendation/images/*.jpg')

# identiy which path we should get (because not all the image paths are what we need)
# store in dictionary
    img_fs_comm = {}
    for f in img_fs:
        for i in imgs_amazon_comm:
            if i in f:
                img_fs_comm[i] = f

# loads the imgs
    imgs = {}
    for f in img_fs_comm.keys():
      #  img = cv2.imread(img_fs_comm[f])
        img = load_img(img_fs_comm[f], grayscale=True, target_size=(224,224))

    # ----- perform similar preprocessing you have
    # gray scale: remove 3rd channel
    #    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img_to_array(img)
    # normalize to 0 and 255 (don't do 0 and 1 because the histogram comparison method need 0 and 255)
    # The way I do will handle the different in skin color better and we focus more on the structure
    # of the skin
        img = (img.max() - img) / (img.max() - img.min()) * 255

    # resize to (224,224)
    # you can uncomment the imshow to take a look on an image
#        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
       # img = cv2.resize(img, (224, 224))

    # plt.imshow(img, cmap='gray')

    # assign
        imgs[f] = img

# ===== preproces the test image so that I can compare with the database image
#    img_test = load_img(image,grayscale=True, target_size=(224, 224))

# ----- perform similar preprocessing you have
# gray scale: remove 3rd channel
    image = Image.open(uploaded_file)
#    img_test = Image.convert(mode="L")
#    img_test = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_test = img_to_array(image)
# normalize to 0 and 255 (don't do 0 and 1 because the histogram comparison method need 0 and 255)
# The way I do will handle the different in skin color better and we focus more on the structure
# of the skin
    img_test = (img_test.max()-img_test)/(img_test.max()-img_test.min())*255

# resize to (224,224)
# you can uncomment the imshow to take a look on an image
#    img_test = cv2.resize(img_test, (224,224), interpolation = cv2.INTER_LINEAR)
#    img_test = cv2.resize(img, (224, 224))
#    img_test = np.asanyarray(img_test)

    df_target = df_new_comm[df_new_comm['skin_type']==classes[0]]


    # Get the distance between the test image and all the other images
    dists = {}
    for imgf in tqdm(df_target.index):
        # get the database image
        img = imgs[imgf]

        # Compare the histogram of both the test image and the database image
        # Return a distance (similarity), the smaller the better
        # You can look at the histogram if you want too
        dist, img1_hist, img2_hist = compare_similarity(img_test, img)


        # append
        dists[imgf] = dist

    # sort the dict by value; return a list of tuple
        dists_sorted = sorted(dists.items(), key=lambda kv: kv[1])

    # output the top-3 working product to match your skin
    k = 3
    df_topk = []
    for dist in dists_sorted:
        # Get the image-associated product
        df_tmp = df_target.loc[dist[0]]

        # if the product is good enough, then store the product
        # score have to be at least 4.0
        if df_tmp['rating_numerical'] > 4.0:
            df_topk.append(df_tmp)

        # break if we have top three
        if len(df_topk) >= k:
            break
    df_topk = pd.DataFrame(df_topk)
    product_name = df_topk['product_name'].tolist()[0]
    product_name2 = df_topk['product_name'].tolist()[1]
    product_name3 = df_topk['product_name'].tolist()[2]
    product_link = df_topk['product_link'].tolist()[0]
    product_link2 = df_topk['product_link'].tolist()[1]
    product_link3 = df_topk['product_link'].tolist()[2]
#    product_link = df_target['link'].tolist()[0]
#    product_name2 = df_target['product_name'].tolist()[1]
#    product_link2 = product_list['link'].tolist()[1]
#    product_name3 = product_list['product_name'].tolist()[2]
#    product_link3 = product_list['link'].tolist()[2]
#    product_image = product_list['product_img'][0]
    st.write("")
    st.write("Recommended products:")
    st.write(product_name)
    st.write(product_link)
    st.write(product_name2)
    st.write(product_link2)
    st.write(product_name3)
    st.write(product_link3)
 #   st.image(product_image)






