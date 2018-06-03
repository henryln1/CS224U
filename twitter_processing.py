import pandas as pd   
import random



twitter_csv_name = 'twitter_gender_classifier_dataset.csv'



def read_csv_get_tweet_gender(csv_file):
	'''
	takes the csv file and processes it into a dictionary that we can use.
	We want a dictionary with two keys
	dict[female] = list of tweets by females
	dict[male] = list of tweets by males

	'''

	gender_tweets_dict = {
		'male': [],
		'female': [],
	}
	csv_df = pd.read_csv(csv_file, encoding = 'latin-1')
	#print(csv_df)
	csv_array = csv_df.as_matrix()
	# print(csv_array[0][5])
	# print(csv_array[0][6])
	# print(csv_array[0][19])

	for row in range(csv_array.shape[0]):
	#for row in range(15):
		gender = csv_array[row][5]
		#print("Row: ", row)
		#print("Gender: ", gender)
		gender_confidence = csv_array[row][6]
		line = csv_array[row][19]
		if float(gender_confidence) == 1:
			if gender in gender_tweets_dict:
				gender_tweets_dict[gender].append(line)

	return gender_tweets_dict

def get_tweets_data(csv_file_name = twitter_csv_name):
	tweets_dict = read_csv_get_tweet_gender(csv_file_name)
	print("Number of male tweets collected: ", len(tweets_dict['male']))
	print("Number of female tweets collected: ", len(tweets_dict['female']))
	lower_number_tweets = len(min(tweets_dict['male'], tweets_dict['female']))
	male_tweets = random.sample(tweets_dict['male'], lower_number_tweets // 2)
	female_tweets = random.sample(tweets_dict['female'], lower_number_tweets // 2)

	labels = ['m' for x in range(lower_number_tweets // 2)] + ['f' for x in range(lower_number_tweets // 2)]
	all_tweets = male_tweets + female_tweets
	return all_tweets, labels