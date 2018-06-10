import csv
import pandas as pd  
import random 

# returns dict {topic -> {gender -> post}}
def get_post_data():
    posts = {}
    csv_df = pd.read_csv('../blogtext.csv', encoding = 'latin-1')
    csv_array = csv_df.as_matrix()
    counter = 0
    for row in csv_array:
        topic = row[3].strip()
        if topic not in posts:
            posts[topic] = {'m': [], 'f': []}
        gender = row[1][0]
        post = row[6].strip()
        posts[topic][gender].append(post)
        counter += 1 
    print('Number of blog posts: ', counter)
    print(counter)
    return posts

# returns list of lines and associated genders w equal # lines from each topic
def get_labeled_lines_equal(unprocessed):
    posts = []
    genders = []
    counter = 0
    all_male = []
    all_female = []
    n = 10000
    for topic, data in unprocessed.items():
        male_posts = data['m']
        female_posts = data['f']
        num_male = len(male_posts)
        num_fem = len(female_posts)
        num_convos = min([num_male, num_fem])
        male_equal = random.sample(male_posts, num_convos)
        female_equal = random.sample(female_posts, num_convos)
        all_male += male_equal
        all_female += female_equal
        counter += num_convos * 2
    posts += random.sample(all_male, n)
    posts += random.sample(all_female, n)
    genders += ['m' if i < n else 'f' for i in range(2 * n)]
    return posts, genders

def get_blog_data():
    posts_unprocessed = get_post_data()
    return get_labeled_lines_equal(posts_unprocessed)

#get_labeled_lines_equal(get_post_data())