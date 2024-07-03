# VideoRetrieval
## GEARS Program: Mission 1
Method: 
  Use Bert to obtain the embedding vector of both the question and the information(main title and subtitles of videos), and check the similarity throught calculating the cosine similarity between them. During training, I label every pair of question and video info as positive(namely max similarity) to train the model,although this may to some degree cause the problem of Sample imbalance.
  Given a question, the whole video infomation list will be iterated and the one having the highest cosine similarity with the question will be chosen as the retrieval result.

### Because of the instability of my VPN, I only download the main title of each video, so the final result is not that precise due to the lack of details(subtitles). I will later ask for help from other teammates to obtian the complete data.
