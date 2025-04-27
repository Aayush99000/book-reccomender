import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix 
from sklearn.neighbors import NearestNeighbors

books = pd.read_csv('books_data/books.csv', sep=";" , encoding="latin-1", on_bad_lines='skip' , low_memory= False)
# print(books.shape)
# print(books.head(2))
# print(books.columns)
books = books[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher',
       'Image-URL-S']]

books.rename(columns={
    "Book-Title":"Title",
    "Book-Author":"Author",
    "Year-Of-Publication":"year",
    "Publisher": "Publisher",
    "Image-URL-S":"url"
} , inplace=True)

# print(books.head(1))

users = pd.read_csv('books_data/users.csv', sep=";" , encoding="latin-1", on_bad_lines='skip' , low_memory= False)
# print(users.head(2))
# print(users.shape)

ratings = pd.read_csv('books_data/ratings.csv', sep=";" , encoding="latin-1", on_bad_lines='skip' , low_memory= False)
# print(ratings.head(2))
# print(ratings.shape)

ratings.rename(columns={
    "User-ID":"user_id",
    "Book-Rating":"rating"
},inplace=True)

print(books.shape)
print(users.shape)
print(ratings.shape)
print(books.head(2))

print(ratings['user_id'].unique().shape)

x= ratings["user_id"].value_counts() >200
print(x[x].shape)

y= x[x].index
print(y)


print(ratings[ratings['user_id'].isin(y)])

#merging the books and ratings 
ratings_with_books = ratings.merge(books, on="ISBN")
print(ratings_with_books)

num_rating = ratings_with_books.groupby('Title')['rating'].count().reset_index()
print(num_rating)

num_rating.rename(columns={"rating": "num_of_rating"}, inplace=True)
print(num_rating)

final_rating= ratings_with_books.merge(num_rating , on='Title')
print(final_rating.head(2))


final_rating= final_rating[final_rating['num_of_rating']>50]

final_rating.drop_duplicates(['user_id','Title'], inplace=True)
print(final_rating.shape)

book_pivot =final_rating.pivot_table(columns='user_id' , index='Title' , values='rating')

book_pivot.fillna(0, inplace=True)
print(book_pivot)

book_sparce = csr_matrix(book_pivot)

#model 
model = NearestNeighbors(algorithm='brute')

model.fit(book_sparce)
distance , suggestion = model.kneighbors(book_pivot.iloc[237,:].values.reshape(-1,1),n_neighbors=6)

print(distance)