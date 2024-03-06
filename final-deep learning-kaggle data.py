#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


kaggle_path = "/Users/huanzhang/Desktop/Computer_Science/final project/data/kaggle/"


# In[3]:


# load data into dataframes 
kaggle_df = pd.read_csv(kaggle_path+"train.csv", sep = ",", header=0)
random_seed = 42


# In[4]:


kaggle_df = kaggle_df.dropna()
kaggle_df.head()


# In[5]:


print(f"There are in total {kaggle_df.shape[0]} records.")
print("Here are some sample text:")
print(kaggle_df['text'][:3])

# Use the apply function with len to find the length of each text in the 'TextColumn'
kaggle_df['text_length'] = kaggle_df['text'].apply(len)

# Calculate and print the longest, shortest, and average length
max_length = kaggle_df['text_length'].max()
min_length = kaggle_df['text_length'].min()
avg_length = kaggle_df['text_length'].mean()

print(f"Longest text has: {max_length} words.")
print(f"Shortest text has: {min_length}")
print(f"Average length of the text are: {round(avg_length)}")


# In[6]:


resiliency_df = kaggle_df[kaggle_df['label'] == 1].head(10)
indices_to_remove = kaggle_df[kaggle_df['label'] == 1].head(10).index
# Remove the records from the original DataFrame
kaggle_df = kaggle_df.drop(indices_to_remove)


# In[7]:


import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Get English stop words from NLTK
stop_words = set(stopwords.words('english'))

# Function to remove stop words and punctuation from a text
def process_text(text):
       # Check if the value is a non-null string
    if isinstance(text, str):
        # Tokenize the text
        tokens = word_tokenize(text)
        
        # Remove stop words
        tokens = [word.lower() for word in tokens if word.lower() not in stop_words]
        
        # Create a Porter Stemmer instance
        porter_stemmer = PorterStemmer()
        
        # Remove punctuation and apply stemming
        tokens = [porter_stemmer.stem(word) for word in tokens]
        
        # Remove punctuation 
        tokens = [word for word in tokens if word not in string.punctuation]
        
        # Join the tokens back into a string
        processed_text = ' '.join(tokens)
        
        return processed_text
    else:
        # If the value is not a string, return an empty string
        return ''
processing_text_test = kaggle_df['text'][0:1]
print(processing_text_test)
processed_text = processing_text_test.apply(process_text)
print(processed_text)


# In[8]:


kaggle_X = kaggle_df['text'].apply(process_text)
kaggle_X = [str(item) for item in kaggle_X]
kaggle_y = np.array(kaggle_df['label'])


# In[9]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

text_corpus = " ".join(kaggle_X)
 
# Generate a word cloud
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_corpus)

# Display the generated word cloud using matplotlib
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")  # Turn off axis labels
plt.show()


# In[10]:


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Vectorize the text data
#vectorizer = CountVectorizer()
vectorizer = TfidfVectorizer()
vectorized_X = vectorizer.fit_transform(kaggle_X)

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(vectorized_X, kaggle_y, test_size=0.2, random_state=random_seed)


# Train the Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = clf.predict(X_valid)

# Evaluate the model

# Perform cross-validation
cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy: {np.mean(cv_scores):.2f}")


# Print confusion matrix and classification report
conf_matrix = confusion_matrix(y_valid, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

class_report = classification_report(y_valid, y_pred)
print("\nClassification Report:")
print(class_report)


# In[11]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalMaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split

# Tokenize the text
tokenizer = Tokenizer(oov_token='<OOV>',  
                      filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                      lower=True,
                      split=' ')
tokenizer.fit_on_texts(kaggle_X)
word_index = tokenizer.word_index

# Convert text to sequences and pad sequences
sequences = tokenizer.texts_to_sequences(kaggle_X)
padded_sequences = pad_sequences(sequences, padding='post', truncating='post')

input_length = max(len(seq) for seq in sequences)

vocabulary_size = len(word_index) + 1

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, kaggle_y, test_size=0.2, random_state=random_seed)

# Build a neural network for text classification
kaggle_model = Sequential()
kaggle_model.add(Embedding(input_dim=vocabulary_size, output_dim=150, input_length=input_length))
kaggle_model.add(GlobalMaxPooling1D()) 
kaggle_model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
kaggle_model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
kaggle_model.add(Dense(1, activation='sigmoid'))

kaggle_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
history = kaggle_model.fit(X_train, 
                    y_train,
                    epochs=10, 
                    batch_size=128,
                    validation_split=0.2)


# In[12]:


import matplotlib.pyplot as plt
# Plot training history
plt.figure(figsize=(12, 6))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()


# In[13]:


from sklearn.metrics import confusion_matrix, classification_report

# Evaluate the model on the test set
test_loss, test_accuracy = kaggle_model.evaluate(X_test, y_test)
print("\nTest accuracy:")
print(test_accuracy)

# Evaluate the model on the test set
y_pred = kaggle_model.predict(X_test)
y_pred_binary = np.round(y_pred)  # Convert probabilities to binary predictions (0 or 1)

# Print confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_binary)
print("\nConfusion Matrix:")
print(conf_matrix)

# Print classification report
class_report = classification_report(y_test, y_pred_binary)
print("\nClassification Report:")
print(class_report)


# In[15]:


resiliency_df.to_csv(kaggle_path+'resiliency.csv')
resiliency_y = np.ones(10)
modified_resiliency_y = np.ones(8)


# In[16]:


modified_1 = """House Dem Aide: We Didn’t Even See Comey’s Letter Until Jason Chaffetz Tweeted It
By Darrell Lucus on October 30, 2016
Subscribe Jason Chaffetz on the stump in American Fork, Utah ( image courtesy Michael Jolley, available under a Creative Commons-BY license)
With apologies to Keith Olbermann, there is no doubt who the Worst Person in The World is this week–FBI Director James Comey. But according to a House Democratic aide, it looks like we also know who the second-worst person is as well. It turns out that when Comey sent his now-infamous letter announcing that the FBI was looking into emails that may be related to Hillary Clinton’s email server, the ranking Democrats on the relevant committees didn’t hear about it from Comey. They found out via a tweet from one of the Republican committee chairmen.
As we now know, Comey notified the Republican chairmen and Democratic ranking members of the House Intelligence, Judiciary, and Oversight committees that his agency was reviewing emails it had recently discovered in order to see if they contained classified information. Not long after this letter went out, Oversight Committee Chairman Jason Chaffetz set the political world ablaze with this tweet.
FBI Dir just informed me, "The FBI has learned of the existence of emails that appear to be pertinent to the investigation." Case reopened
— Jason Chaffetz (@jasoninthehouse) October 28, 2016
Of course, we now know that this was not the case. Comey was actually saying that it was reviewing the emails in light of “an unrelated case”–which we now know to be Anthony Weiner’s sexting with a teenager. But apparently such little things as facts didn’t matter to Chaffetz. The Utah Republican had already vowed to initiate a raft of investigations if Hillary wins–at least two years’ worth, and possibly an entire term’s worth of them. Apparently Chaffetz thought the FBI was already doing his work for him–resulting in a tweet that briefly roiled the nation before cooler heads realized it was a dud.
But according to a senior House Democratic aide, misreading that letter may have been the least of Chaffetz’ sins. That aide told Shareblue that his boss and other Democrats didn’t even know about Comey’s letter at the time–and only found out when they checked Twitter.
“Democratic Ranking Members on the relevant committees didn’t receive Comey’s letter until after the Republican Chairmen. In fact, the Democratic Ranking Members didn’ receive it until after the Chairman of the Oversight and Government Reform Committee, Jason Chaffetz, tweeted it out and made it public.”
So let’s see if we’ve got this right. The FBI director tells Chaffetz and other GOP committee chairmen about a major development in a potentially politically explosive investigation, and neither Chaffetz nor his other colleagues had the courtesy to let their Democratic counterparts know about it. Instead, according to this aide, he made them find out about it on Twitter.
There has already been talk on Daily Kos that Comey himself provided advance notice of this letter to Chaffetz and other Republicans, giving them time to turn on the spin machine. That may make for good theater, but there is nothing so far that even suggests this is the case. After all, there is nothing so far that suggests that Comey was anything other than grossly incompetent and tone-deaf.
What it does suggest, however, is that Chaffetz is acting in a way that makes Dan Burton and Darrell Issa look like models of responsibility and bipartisanship. He didn’t even have the decency to notify ranking member Elijah Cummings about something this explosive. If that doesn’t trample on basic standards of fairness, I don’t know what does.
Granted, it’s not likely that Chaffetz will have to answer for this. He sits in a ridiculously Republican district anchored in Provo and Orem; it has a Cook Partisan Voting Index of R+25, and gave Mitt Romney a punishing 78 percent of the vote in 2012. Moreover, the Republican House leadership has given its full support to Chaffetz’ planned fishing expedition. But that doesn’t mean we can’t turn the hot lights on him. After all, he is a textbook example of what the House has become under Republican control. And he is also the Second Worst Person in the World.
About Darrell Lucus
Darrell is a 30-something graduate of the University of North Carolina who considers himself a journalist of the old school. An attempt to turn him into a member of the religious right in college only succeeded in turning him into the religious right's worst nightmare--a charismatic Christian who is an unapologetic liberal. His desire to stand up for those who have been scared into silence only increased when he survived an abusive three-year marriage. You may know him on Daily Kos as Christian Dem in NC. Follow him on Twitter @DarrellLucus or connect with him on Facebook. Click here to buy Darrell a Mello Yello. Connect"""

modified_2 = """Why the Truth Might Get You Fired
October 29, 2016
The tension between intelligence analysts and political policymakers has always been between honest assessments and desired results, with the latter often overwhelming the former, as in the Iraq War, writes Lawrence Davidson.
By Lawrence Davidson
For those who might wonder why foreign policy makers repeatedly make bad choices, some insight might be drawn from the following analysis. The action here plays out in the United States, but the lessons are probably universal.
Back in the early spring of 2003, George W. Bush initiated the invasion of Iraq. One of his key public reasons for doing so was the claim that the country’s dictator, Saddam Hussein, was on the verge of developing nuclear weapons and was hiding other weapons of mass destruction. The real reason went beyond that charge and included a long-range plan for “regime change” in the Middle East. President George W. Bush and Vice President Dick Cheney receive an Oval Office briefing from CIA Director George Tenet. Also present is Chief of Staff Andy Card (on right). (White House photo)
For our purposes, we will concentrate on the belief that Iraq was about to become a hostile nuclear power. Why did President Bush and his close associates accept this scenario so readily?
The short answer is Bush wanted, indeed needed, to believe it as a rationale for invading Iraq. At first, he had tried to connect Saddam Hussein to the 9/11 attacks on the U.S. Though he never gave up on that stratagem, the lack of evidence made it difficult to rally an American people, already fixated on Afghanistan, to support a war against Baghdad.
But the nuclear weapons gambit proved more fruitful, not because there was any hard evidence for the charge, but because supposedly reliable witnesses, in the persons of exiled anti-Saddam Iraqis (many on the U.S. government’s payroll ), kept telling Bush and his advisers that the nuclear story was true.
What we had was a U.S. leadership cadre whose worldview literally demanded a mortally dangerous Iraq, and informants who, in order to precipitate the overthrow of Saddam, were willing to tell the tale of pending atomic weapons. The strong desire to believe the tale of a nuclear Iraq lowered the threshold for proof . Likewise, the repeated assertions by assumed dependable Iraqi sources underpinned a nationwide U.S. campaign generating both fear and war fever.
So the U.S. and its allies insisted that the United Nations send in weapons inspectors to scour Iraq for evidence of a nuclear weapons program (as well as chemical and biological weapons). That the inspectors could find no convincing evidence only frustrated the Bush administration and soon forced its hand.
On March 19, 2003, Bush launched the invasion of Iraq with the expectation was that, once in occupation of the country, U.S. inspectors would surely find evidence of those nukes (or at least stockpiles of chemical and biological weapons). They did not. Their Iraqi informants had systematically lied to them.
Social and Behavioral Sciences to the Rescue?
The various U.S. intelligence agencies were thoroughly shaken by this affair, and today, 13 years later, their directors and managers are still trying to sort it out – specifically, how to tell when they are getting “true” intelligence and when they are being lied to. Or, as one intelligence worker has put it, we need “ help to protect us against armies of snake oil salesmen. ” To that end the CIA et al. are in the market for academic assistance.
Ahmed Chalabi, head of the Iraqi National Congress, a key supplier of Iraqi defectors with bogus stories of hidden WMD.
A “partnership” is being forged between the Office of the Director of National Intelligence (ODNI), which serves as the coordinating center for the sixteen independent U.S. intelligence agencies, and the National Academies of Sciences, Engineering and Medicine . The result of this collaboration will be a “ permanent Intelligence Community Studies Board” to coordinate programs in “social and behavioral science research [that] might strengthen national security .”
Despite this effort, it is almost certain that the “social and behavioral sciences” cannot give the spy agencies what they want – a way of detecting lies that is better than their present standard procedures of polygraph tests and interrogations. But even if they could, it might well make no difference, because the real problem is not to be found with the liars. It is to be found with the believers.
The Believers
It is simply not true, as the ODNI leaders seem to assert, that U.S. intelligence agency personnel cannot tell, more often than not, that they are being lied to. This is the case because there are thousands of middle-echelon intelligence workers, desk officers, and specialists who know something closely approaching the truth – that is, they know pretty well what is going on in places like Afghanistan, Iraq, Syria, Libya, Israel, Palestine and elsewhere.
Director of National Intelligence James Clapper (right) talks with President Barack Obama in the Oval Office, with John Brennan and other national security aides present. (Photo credit: Office of Director of National Intelligence)
Therefore, if someone feeds them “snake oil,” they usually know it. However, having an accurate grasp of things is often to no avail because their superiors – those who got their appointments by accepting a pre-structured worldview – have different criterion for what is “true” than do the analysts.
Listen to Charles Gaukel, of the National Intelligence Council – yet another organization that acts as a meeting ground for the 16 intelligence agencies. Referring to the search for a way to avoid getting taken in by lies, Gaukel has declared, “ We’re looking for truth. But we’re particularly looking for truth that works. ” Now what might that mean?
I can certainly tell you what it means historically. It means that for the power brokers, “truth” must match up, fit with, their worldview – their political and ideological precepts. If it does not fit, it does not “work.” So the intelligence specialists who send their usually accurate assessments up the line to the policy makers often hit a roadblock caused by “group think,” ideological blinkers, and a “we know better” attitude.
On the other hand, as long as what you’re selling the leadership matches up with what they want to believe, you can peddle them anything: imaginary Iraqi nukes, Israel as a Western-style democracy, Saudi Arabia as an indispensable ally, Libya as a liberated country, Bashar al-Assad as the real roadblock to peace in Syria, the Strategic Defense Initiative (SDI) aka Star Wars, a world that is getting colder and not warmer, American exceptionalism in all its glory – the list is almost endless.
What does this sad tale tell us? If you want to spend millions of dollars on social and behavioral science research to improve the assessment and use of intelligence, forget about the liars. What you want to look for is an antidote to the narrow-mindedness of the believers – the policymakers who seem not to be able to rise above the ideological presumptions of their class – presumptions that underpin their self-confidence as they lead us all down slippery slopes.
It has happened this way so often, and in so many places, that it is the source of Shakespeare’s determination that “what is past, is prelude.” Our elites play out our destinies as if they have no free will – no capacity to break with structured ways of seeing. Yet the middle-echelon specialists keep sending their relatively accurate assessments up the ladder of power. Hope springs eternal."""

modified_3 = """Videos: 15 Civilians Killed In Single US Airstrike Have Been Identified
The rate at which civilians are being killed by American airstrikes in Afghanistan is now higher than it was in 2014 when the US was engaged in active combat operations. Photo of Hellfire missiles being loaded onto a US military Reaper drone in Afghanistan by Staff Sgt. Brian Ferguson/U.S. Air Force.
The Bureau has been able to identify 15 civilians killed in a single US drone strike in Afghanistan last month – the biggest loss of civilian life in one strike since the attack on the Medecins Sans Frontieres hospital (MSF) last October.
The US claimed it had conducted a “counter-terrorism” strike against Islamic State (IS) fighters when it hit Nangarhar province with missiles on September 28. But the next day the United Nations issued an unusually rapid and strong statement saying the strike had killed 15 civilians and injured 13 others who had gathered at a house to celebrate a tribal elder’s return from a pilgrimage to Mecca.
The Bureau spoke to a man named Haji Rais who said he was the owner of the house that was targeted. He said 15 people were killed and 19 others injured, and provided their names (listed below). The Bureau was able to independently verify the identities of those who died.
Rais’ son, a headmaster at a local school, was among them. Another man, Abdul Hakim, lost three of his sons in the attack.
Rais said he had no involvement with IS and denied US claims that IS members had visited his house before the strike. He said: “I did not even speak to those sort of people on the phone let alone receiving them in my house.”
The deaths amount to the biggest confirmed loss of civilian life in a single American strike in Afghanistan since the attack on the MSF hospital in Kunduz last October, which killed at least 42 people.
The Nangarhar strike was not the only US attack to kill civilians in September. The Bureau’s data indicates that as many as 45 civilians and allied soldiers were killed in four American strikes in Afghanistan and Somalia that month.
On September 18 a pair of strikes killed eight Afghan policemen in Tarinkot, the capital of Urozgan province. US jets reportedly hit a police checkpoint, killing one officer, before returning to target first responders. The use of this tactic – known as a “double-tap” strike – is controversial because they often hit civilian rescuers.
The US told the Bureau it had conducted the strike against individuals firing on and posing a threat to Afghan forces. The email did not directly address the allegations of Afghan policemen being killed.
At the end of the month in Somalia, citizens burnt US flags on the streets of the north-central city of Galcayo after it emerged a drone attack may have unintentionally killed 22 Somali soldiers and civilians. The strike occurred on the same day as the one in Nangarhar.
In both the Somali and Afghan incidents, the US at first denied that any non-combatants had been killed. It is now investigating both the strikes in Nangarhar and Galcayo.
The rate at which civilians are being killed by American airstrikes in Afghanistan is now higher than it was in 2014 when the US was engaged in active combat operations."""

modified_4 = """An Iranian woman, Golrokh Ebrahimi Iraee, 35, has received a six-year prison sentence following a search by Iran’s Revolutionary Guard. The search uncovered a notebook containing a fictional story she had written depicting the stoning of a woman to death, as reported by Eurasia Review.
Golrokh Ebrahimi Iraee is the wife of political prisoner Arash Sadeghi, 36, currently serving a 19-year sentence for his role as a human rights activist.
The incident unfolded when the intelligence unit of the Revolutionary Guards, without a warrant, raided their apartment during the arrest of her husband. During the raid, drafts of stories written by Ebrahimi Iraee were discovered.
One of the seized drafts portrayed the stoning of women to death for adultery, a narrative that was never published or presented to anyone. The story followed the protagonist who watched a movie depicting the stoning of women under Islamic law for adultery."""

modified_5 = """The enigma surrounding The Third Reich and Nazi Germany remains a topic of discussion among various observers. Some assert that under Adolf Hitler's rule, Nazi Germany possessed supernatural powers and heavily relied on pseudo-science from 1933 to 1945. However, others argue that such claims are mere speculation without factual basis. Over the years, researchers have delved into the mysteries linked to Nazi Germany.
In 1941, Nazi Germany invaded Russia (formerly the USSR) during the Second World War. The German army advanced deep into Russian territory, nearing Moscow, before facing a counter-attack from the Russians, eventually pushing the Nazis back.
Amid the Nazi occupation in 1942, a clandestine military base, codenamed "Schatzgraber" or "Treasure Hunter," emerged in the Arctic. Reportedly crucial in the war against Russia, the base served as a tactical weather station for planning strategic movements of Nazi troops, warships, and submarines. It also housed prominent Nazi scientists who conducted experiments to aid Germany's war efforts. Speculation arose that the Nazis used the base to communicate with aliens or extraterrestrial beings, with controversial ties to the Ahnenerbe, an institute in Nazi Germany known for researching the Aryan race's archaeological and cultural history, rumored to have occult influences.
However, the Nazis abandoned the base in 1944, coinciding with the Russian army's offensive that expelled the Germans from the country. According to wartime accounts, dwindling supplies forced Nazi officers at the base to resort to consuming polar bear meat, leading to severe illness from trichinosis. This compelled a German U-boat rescue. While some consider the story of "Treasure Hunter" a myth, Russian researchers now claim its discovery on the island of Alexandra Land in the Arctic Circle, 620 miles from the North Pole.
Senior researcher at the Russian Arctic National Park, Evgeny Ermolov, stated, "Before it was only known from written sources, but now we also have real proof."""

modified_6 = """Clinton Campaign Urges FBI to Confirm Trump's Russia Ties
As the 2016 election campaign approaches its conclusion, the Clinton campaign is intensifying its calls for the FBI to publicly validate the campaign's assertions that Republican nominee Donald Trump is covertly collaborating with Russia. Senator Harry Reid (D – NV) has gone to the extent of alleging that the FBI possesses undisclosed "explosive" evidence of coordination between the Trump campaign and the Russian government, which it is withholding.
FBI officials, well-versed in their investigations into these allegations, initiated by the Clinton campaign around the Democratic National Convention, maintain that they have found no evidence linking Trump and Russia. This has led FBI Director James Comey to refrain from making any statements supporting such claims.
Persisting with these allegations for an extended period, the Clinton campaign now asserts that "everyone knows" they are true. They seem disconcerted by the FBI's reluctance to endorse the claims, emphasizing the lack of real evidence supporting the narrative.
Despite repeated denials of ties to Russia by the Trump campaign, Clinton labeled Trump a "puppet" of Russian President Vladimir Putin during the final presidential debate. The demands for affirmation have intensified since the FBI's report to Congress on the pursuit of additional Clinton emails last Friday.
As Clinton's primary campaign scandal escalates in the final weeks, some within her campaign suggest that confirming Trump's secret collaboration with the Russians would be equitable. However, in the absence of evidence, such an affirmation seems unlikely to occur."""

modified_7 = """UN Committee Votes to Initiate Nuclear Weapons Ban Treaty Negotiations
In a historic move, the United Nations First Committee voted on Thursday to convene a conference in March for negotiating a new treaty to ban the possession of nuclear weapons. This significant vote marks a substantial advancement in the global campaign initiated by non-nuclear weapons states and civil society to rid the world of nuclear weapons.
Expressing dismay over the failure of nuclear weapons states to fulfill their obligation under Article VI of the Non-Proliferation Treaty, which mandates good faith negotiations for the elimination of their nuclear arsenals, and fueled by the escalating danger of nuclear war, over 120 nations convened in Oslo in March 2013. The focus shifted from abstract considerations of nuclear strategy to an evaluation of the medical data illustrating the catastrophic consequences of using nuclear weapons. Notably, all major nuclear powers, the US, Russia, UK, China, and France, boycotted the conference.
Subsequent meetings in Nayarit, Mexico, and Vienna in 2014 led to a pledge by the Austrian government to "close the gap" in international law that currently lacks a specific prohibition on the possession of nuclear weapons. Over 140 countries associated themselves with the pledge, vehemently opposed by the US and other nuclear weapons states. In the fall of 2015, the UN General Assembly established an Open-Ended Working Group, which met in Geneva earlier this year and recommended the negotiations approved on Thursday.
Despite US hopes to limit the positive vote to less than one hundred, the final tally was 123 in favor, 38 against, and 16 abstentions. The negative votes came from nuclear weapons states and US allies in NATO, along with Japan, South Korea, and Australia, considering themselves under the protection of the "US nuclear umbrella."
Breaking ranks, four nuclear weapons states, China, India, and Pakistan abstained, and North Korea voted in favor of the treaty negotiations. Additionally, the Netherlands and Finland defied pressure, abstaining from the vote. Japan, despite voting against the treaty, expressed intent to participate in the negotiations set to begin in March.
While the US and other nuclear weapons states may attempt to block the final approval of the treaty conference by the General Assembly, Thursday's vote strongly indicates that negotiations are likely to commence in March, involving a significant majority of UN member states, even if the nuclear states persist in their boycott.
The successful establishment of a new treaty won't eliminate nuclear weapons outright, but it will exert substantial pressure on nuclear weapons states, emphasizing their reluctance to uphold obligations under the Non-Proliferation Treaty while demanding compliance from non-nuclear weapons states.
In light of the numerous close calls with nuclear war over the past 70 years, there is an urgent need to join and lead the global movement to abolish nuclear weapons. Efforts should be directed toward bringing all nuclear weapons states into a binding agreement that outlines a detailed timeline for elimination, along with robust verification and enforcement mechanisms to ensure compliance.
While this task is undoubtedly challenging, the consequences of inaction are dire. The decision is ours: eliminate these weapons or face the risk of their use, potentially leading to the destruction of human civilization.
Dr. Ira Helfand, MD, served as the past president of Physicians for Social Responsibility and currently holds the position of co-president of the International Physicians for the Prevention of Nuclear War, the 1985 Nobel Peace Prize recipient."""

modified_8 = """Caddo Nation Tribal Leader Freed After Dakota Access Pipeline Protest Arrest
In a recent development, a tribal leader of the Caddo Nation, Tamara Francis-Fourkiller, has been released after spending two days in custody in North Dakota. Family members assert that she was an innocent bystander caught in a clash between law enforcement and protesters, refuting any allegations made by the police.
Jessi Mitchell, reporting for local News 9, reveals that an anonymous donor contributed $2.5 million on Saturday afternoon to secure the release of everyone arrested on Thursday at the Dakota Access Pipeline site. However, it was emphasized that Francis-Fourkiller was never supposed to be arrested in the first place.
An expert on sacred burial grounds, Francis-Fourkiller, along with other tribal leaders, visited the Sioux of Standing Rock to offer advice during negotiations with the Dakota Access Pipeline construction team. Loretta Francis, the sister of Francis-Fourkiller, explained that the tribal leaders were concerned about the desecration of remains in the pipeline and had gathered for a conference.
During their visit, Francis-Fourkiller and other leaders decided to tour the protest camps, never anticipating their arrest. Francis pointed out that her sister had no access to medication while in custody in Cass County, North Dakota, and is now facing charges of conspiracy and rioting. She expressed the historical context, mentioning the Trail of Tears and the suffering of their family, expressing the desire for a better future for the next generation.
On Saturday afternoon, dozens of Native Americans from Oklahoma tribes gathered at the state Capitol to voice their anger over the treatment of protesters in North Dakota, particularly highlighting the recent acquittal of armed protesters at an Oregon wildlife refuge. Comanche Nation tribal council member Sonya Nevaquaya stated that they are unarmed and decried the use of military force against them, labeling it as shameful.
The protection of land is a fundamental principle for all Native American tribes. Chanting "Water is life!" at the Oklahoma demonstration, participants aimed to rally support nationwide against the pipeline project. Concerns were raised about the potential environmental impact of pipelines on water resources. Francis-Fourkiller stated her intention to return to her home in Norman as soon as possible."""


# In[17]:


resiliency_X = resiliency_df['text']

modified_resiliency_X = [modified_1, 
                         modified_2,
                         modified_3, 
                         modified_4, 
                         modified_5, 
                         modified_6, 
                         modified_7,
                         modified_8]


# In[18]:


print(len(resiliency_X), len(resiliency_y))


# In[20]:


# Convert text to sequences and pad sequences
resiliency_sequences = tokenizer.texts_to_sequences(resiliency_X)
resiliency_padded_sequences = pad_sequences(resiliency_sequences, padding='post', truncating='post')
# Evaluate the model on the test set
y_pred = kaggle_model.predict(resiliency_padded_sequences)
y_pred_binary = np.round(y_pred)  # Convert probabilities to binary predictions (0 or 1)

# Print confusion matrix
conf_matrix = confusion_matrix(resiliency_y, y_pred_binary)
print("\nConfusion Matrix:")
print(conf_matrix)

# Print classification report
class_report = classification_report(resiliency_y, y_pred_binary)
print("\nClassification Report:")
print(class_report)


# In[22]:


# Convert text to sequences and pad sequences
modified_resiliency_sequences = tokenizer.texts_to_sequences(modified_resiliency_X)
modified_resiliency_padded_sequences = pad_sequences(modified_resiliency_sequences, padding='post', truncating='post')
# Evaluate the model on the test set
y_pred = kaggle_model.predict(modified_resiliency_padded_sequences)
y_pred_binary = np.round(y_pred)  # Convert probabilities to binary predictions (0 or 1)

# Print confusion matrix
conf_matrix = confusion_matrix(modified_resiliency_y, y_pred_binary)
print("\nConfusion Matrix:")
print(conf_matrix)

# Print classification report
class_report = classification_report(modified_resiliency_y, y_pred_binary)
print("\nClassification Report:")
print(class_report)

