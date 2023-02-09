import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
import string
from sklearn import metrics
from nltk.corpus import stopwords 
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import tkinter as tk


sns.set()

def clean(text):
    stop_words=set(stopwords.words("arabic"))
    unique_words = []
    words = text.split()

    for word in words:
        if ((word not in unique_words) and (word not in stop_words) and (word not in string.punctuation)) :
                unique_words.append(word)
    
    return " ".join(unique_words)

def read_tsv(data_file):
    text_data = list()
    labels = list()
    infile = open(data_file, encoding='utf-8')
    
    for line in infile:
        if not line.strip():
            continue
        label, text = line.split('\t')
        text_data.append(clean(text))
        labels.append(label)
    
    return text_data,labels

def load(pos_file,neg_file):
    pos_data, pos_labels = read_tsv(pos_file)
    neg_data, neg_labels = read_tsv(neg_file)

    pos_train_data, pos_test_data,pos_train_label, pos_test_label = train_test_split(pos_data,pos_labels,train_size=0.75, shuffle=True)
    neg_train_data, neg_test_data,neg_train_label, neg_test_label = train_test_split(neg_data,neg_labels,train_size=0.75, shuffle=True)

    x_train = pos_train_data + neg_train_data
    y_train = pos_train_label + neg_train_label

    x_test = pos_test_data + neg_test_data
    y_test = pos_test_label + neg_test_label

    print('train data size:{}\ttest data size:{}'.format(len(y_train), len(y_test)))
    print('train data: number of pos:{}\tnumber of neg:{}\t'.format(y_train.count('pos'), y_train.count('neg')))
    print('test data: number of pos:{}\tnumber of neg:{}\t'.format(y_test.count('pos'), y_test.count('neg')))
    print('------------------------------------')
    return x_train, y_train, x_test, y_test

def my_predictions(my_sentence, model):
    return  model.predict([my_sentence])

pos = 'Positive+Tweets.tsv'
neg = 'Negative+Tweets.tsv'

x_train, y_train, x_test, y_test =load(pos,neg)

model_NeuralNetworks = make_pipeline(TfidfVectorizer(encoding= "utf-8"), 
                                    MLPClassifier(learning_rate='adaptive', 
                                    hidden_layer_sizes=(10,5), max_iter=10000,solver='lbfgs'))
# Train the model using the training data
model_NeuralNetworks.fit(x_train, y_train)
# Predict the categories of the test data
y_predicted_NeuralNetwork = model_NeuralNetworks.predict(x_test)



model_naivebaiase = make_pipeline(TfidfVectorizer(encoding= "utf-8"), MultinomialNB())
# Train the model using the training data
model_naivebaiase.fit(x_train, y_train)
# Predict the categories of the test data
y_predicted_naivebiase = model_naivebaiase.predict(x_test)


model_DecisionTree = make_pipeline(TfidfVectorizer(encoding= "utf-8"),  DecisionTreeClassifier())
# Train the model using the training data
model_DecisionTree.fit(x_train,y_train)
# Predict the categories of the test data
y_predicted_DecisionTree = model_DecisionTree.predict(x_test)

print("Naive Bayez\n",metrics.classification_report(y_test, y_predicted_naivebiase,target_names=['pos', 'neg']))
print('------------------------------------')
print("Decision Tree \n",metrics.classification_report(y_test, y_predicted_DecisionTree,target_names=['pos', 'neg']))
print('------------------------------------')
print("Neural Network \n",metrics.classification_report(y_test, y_predicted_NeuralNetwork,target_names=['pos', 'neg']))

mat = confusion_matrix(y_test, y_predicted_naivebiase)
sns.heatmap(mat, square = True, annot=True, fmt = "d")
plt.xlabel("true labels")
plt.ylabel("predicted label")
plt.suptitle("naive bayez")
plt.show()

mat = confusion_matrix(y_test, y_predicted_DecisionTree)
sns.heatmap(mat, square = True, annot=True, fmt = "d")
plt.xlabel("true labels")
plt.ylabel("predicted label")
plt.suptitle("desicion tree")
plt.show()

mat = confusion_matrix(y_test, y_predicted_NeuralNetwork)
sns.heatmap(mat, square = True, annot=True, fmt = "d")
plt.xlabel("true labels")
plt.ylabel("predicted label")
plt.suptitle("nural network")
plt.show()

def nb_cmd():
    # Get the text from the search bar
    search_text = search_var.get()
    # Display a message
    if(my_predictions(search_text, model_naivebaiase)=="pos"):
        message='possitive :)'
    elif (my_predictions(search_text, model_naivebaiase)=="neg"):
        message='negative :('
    result_label.config(text=message)
def dt_cmd():
    # Get the text from the search bar
    search_text = search_var.get()
    # Display a message
    if(my_predictions(search_text, model_DecisionTree)=="pos"):
        message='possitive :)'
    elif (my_predictions(search_text, model_DecisionTree)=="neg"):
        message='negative :('
    result_label.config(text=message)
def nn_cmd():
    # Get the text from the search bar
    search_text = search_var.get()
    # Display a message
    if(my_predictions(search_text, model_NeuralNetworks)=="pos"):
        message='possitive :)'
    elif (my_predictions(search_text, model_NeuralNetworks)=="neg"):
        message='negative :('
    result_label.config(text=message)

# Create the main window
root = tk.Tk()
root.title("predect the emotion ai")
root.geometry("400x400")
root.config(bg='#2c3e50')

# Create a search bar
search_var = tk.StringVar()
search_entry = tk.Entry(root, textvariable=search_var, bg='white')
search_entry.grid(row=1, column=0, padx=10, pady=10, sticky='ew')

label = tk.Label(root, text="wellcome to our emotion predection ai !!", bg='#2c3e50', fg='yellow',font=("Helvetica", 15))
label.grid(row=0, column=0, padx=10, pady=10, sticky='ew')

# Create a label to display the result
result_label = tk.Label(root, text=" ", bg='#2c3e50', fg='yellow',font=("Helvetica", 15))
result_label.grid(row=2, column=0, padx=10, pady=10, sticky='ew')

# Create a naivebiase button
nb_button = tk.Button(root, text="predict using naive bayez", command=nb_cmd, bg='#f1c40f', activebackground='#f7dc6f')
nb_button.grid(row=3, column=0, padx=10, pady=10, sticky='ew')
# Create a desicion tree button
dt_button = tk.Button(root, text="predict using desicion tree", command=dt_cmd, bg='#f1c40f', activebackground='#f7dc6f')
dt_button.grid(row=4, column=0, padx=10, pady=10, sticky='ew')

# Create a neural network button
dt_button = tk.Button(root, text="predict using neural network", command=nn_cmd, bg='#f1c40f', activebackground='#f7dc6f')
dt_button.grid(row=5, column=0, padx=10, pady=10, sticky='ew')
# Center the search bar and button
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
root.rowconfigure(1, weight=1)
root.rowconfigure(2, weight=1)
root.rowconfigure(3, weight=1)
root.rowconfigure(4, weight=1)
root.rowconfigure(5, weight=1)
# Start the GUI event loop
root.mainloop()