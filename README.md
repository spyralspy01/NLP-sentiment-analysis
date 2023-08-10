Title: Sentiment Analysis of Welsh Tweets Using Emoji-based Proxy Labeling

Surya Shasank Dendukuru
Saint Louis University

Abstract:
Sentiment analysis, a pivotal aspect of Natural Language Processing (NLP), is key to understanding public sentiment and emotions expressed on social media platforms. This project adopts a novel approach to perform sentiment analysis on Welsh tweets. Given the absence of a gold-standard training corpus for Welsh sentiment analysis, this project leverages the presence or absence of "happy" and "sad" emoji/emoticons as a proxy for sentiment classification. The report elaborates on the rationale behind this approach, and its methodology, and presents the project's findings and implications.

1. Introduction:
Sentiment analysis is an essential facet of NLP, aiding in extracting insights from textual data across diverse applications. This project is dedicated to sentiment analysis of tweets written in the Welsh language. Since Welsh lacks a comprehensive labeled dataset for sentiment analysis, we employ a distinctive strategy utilizing emoji/emoticons as a surrogate for sentiment classification.

2. Approach: Emoji-based Proxy Labeling:
The absence of a well-established Welsh sentiment corpus prompted the adoption of an unconventional technique. We leverage the presence or absence of "happy" and "sad" emoji/emoticons as a way to categorize sentiment in tweets. The rationale behind this approach is that emojis often convey emotions succinctly, making them a pragmatic means to infer sentiment even in languages with limited resources.

3. Methodology:
We first collected a sizable dataset of Welsh tweets from various social media platforms to implement emoji-based proxy labeling. We then developed a custom script to determine the presence or absence of "happy" and "sad" emojis. Based on this binary classification, we categorized tweets as positive, neutral, or negative sentiment. Supervised machine learning techniques were employed to train and validate the model.

4. Findings and Implications:
The results of the sentiment analysis reveal exciting insights into the emotional landscape of Welsh tweets. Our approach demonstrates a reasonable accuracy in categorizing sentiment, showing the viability of utilizing emojis for sentiment classification when traditional labeled data is scarce. This approach not only facilitates sentiment analysis in resource-scarce languages but also sheds light on the potential of emoji-emotion correlations in various linguistic contexts.

5. Limitations and Future Directions:
While the emoji-based proxy labeling approach is innovative, it has limitations, such as the potential for misclassification due to the ambiguity of emojis and cultural variations in interpretation. In the future, efforts can be directed towards creating a labeled dataset for Welsh sentiment analysis, incorporating more nuanced linguistic features, and exploring cross-lingual transfer learning techniques.

6. Practical Applications:
The developed sentiment analysis model using emojis holds the potential for social media monitoring, gauging public opinion, and understanding emotional trends in Welsh online discourse. Additionally, the approach paves the way for sentiment analysis in other languages facing similar resource constraints.

7. Conclusion:
This project showcases the innovative application of emoji-based proxy labeling for sentiment analysis in Welsh tweets. The absence of a gold-standard training corpus led to this creative approach, showcasing the adaptability of sentiment analysis techniques to unique linguistic challenges. The findings underscore the potential of emojis as universal emotional markers and open avenues for future research in resource-scarce NLP domains.
