from flask import (
    Flask,
    g,
    redirect,
    render_template,
    request,
    session,
    url_for
)
import itertools
import pandas as pd
import numpy as np
from flask_table import Table, Col
from mlxtend.frequent_patterns import (apriori,association_rules,)

class User:
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

    def __repr__(self):
        return f'<User: {self.username}>'



users = []
users.append(User(id=1, username='miq', password='mar123'))
users.append(User(id=2, username='mom', password='mar678'))
users.append(User(id=3, username='Carlos', password='somethingsimple'))

#building flask table for showing recommendation results
class Results(Table):
    id = Col('Id', show=False)
    title = Col('Recommendation List')


app = Flask(__name__)
app.secret_key = 'somesecretkeythatonlyishouldknow'

df = pd.read_csv("datasets/results/Association_result_for_online_reatil_data.csv")
data=df[['antecedents', 'consequents']]
itemset_count = str(int(df['itemset_count'].iloc[800]))
rules_count = str(int(df['rules_count'].iloc[800]))
items = sorted(list(df['items '].iloc[800:871]))
basket = set()

@app.before_request
def before_request():
    g.user = None

    if 'user_id' in session:
        user = [x for x in users if x.id == session['user_id']][0]
        g.user = user
        

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        session.pop('user_id', None)

        username = request.form['username']
        password = request.form['password']
        
        user = [x for x in users if x.username == username][0]
        if user and user.password == password:
            session['user_id'] = user.id
            return redirect(url_for('profile'))

        else:
            return render_template('login.html',info="Invalid Password!",info1="Contact Admin")

    return render_template('login.html')

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if not g.user:
        return redirect(url_for('login'))
    else:
        if request.method == 'POST':
            if request.form['movie_button'] == 'movie':
                return render_template('welcome.html')
            else:
                return render_template('profile.html')
        else:
            return render_template('profile.html')

#Rating Page
@app.route("/rating", methods=["GET", "POST"])
def rating():
    if request.method=="POST":
        return render_template('recommendation.html')
    return render_template('rating.html')

#Results Page
@app.route("/recommendation", methods=["GET", "POST"])
def recommendation():
    if request.method == 'POST':
        #reading the original dataset
        movies = pd.read_csv('datasets/movies.csv')

        #separating genres for each movie
        movies = pd.concat([movies, movies.genres.str.get_dummies(sep='|')], axis=1)

        #dropping variables to have a dummy 1-0 matrix of movies and their genres
        ## IMAX is not a genre, it is a specific method of filming a movie, thus removed
        ###we do not need movieId for this project
        categories = movies.drop(['title', 'genres', 'IMAX', 'movieId'], axis=1)

        #initializing user preference list which will contain user ratings
        preferences = []

        #reading rating values given by user in the front-end
        Action = request.form.get('Action')
        Adventure = request.form.get('Adventure')
        Animation = request.form.get('Animation')
        Children = request.form.get('Children')
        Comedy = request.form.get('Comedy')
        Crime = request.form.get('Crime')
        Documentary = request.form.get('Documentary')
        Drama = request.form.get('Drama')
        Fantasy = request.form.get('Fantasy')
        FilmNoir = request.form.get('FilmNoir')
        Horror = request.form.get('Horror')
        Musical = request.form.get('Musical')
        Mystery = request.form.get('Mystery')
        Romance = request.form.get('Romance')
        SciFi = request.form.get('SciFi')
        Thriller = request.form.get('Thriller')
        War = request.form.get('War')
        Western = request.form.get('Western')

        #inserting each rating in a specific position based on the movie-genre matrix
        preferences.insert(0, int(Action))
        preferences.insert(1,int(Adventure))
        preferences.insert(2,int(Animation))
        preferences.insert(3,int(Children))
        preferences.insert(4,int(Comedy))
        preferences.insert(5,int(Crime))
        preferences.insert(6,int(Documentary))
        preferences.insert(7,int(Drama))
        preferences.insert(8,int(Fantasy))
        preferences.insert(9,int(FilmNoir))
        preferences.insert(10,int(Horror))
        preferences.insert(11,int(Musical))
        preferences.insert(12,int(Mystery))
        preferences.insert(13,int(Romance))
        preferences.insert(14,int(SciFi))
        preferences.insert(15,int(War))
        preferences.insert(16,int(Thriller))
        preferences.insert(17,int(Western))

        #This funtion will get each movie score based on user's ratings through dot product
        def get_score(a, b):
           return np.dot(a, b)

        #Generating recommendations based on top score movies
        def recommendations(X, n_recommendations):
            movies['score'] = get_score(categories, preferences)
            return movies.sort_values(by=['score'], ascending=False)['title'][:n_recommendations]

        #printing top-20 recommendations
        output= recommendations(preferences, 20)
        table = Results(output)
        table.border = True
        return render_template('recommendation.html', table=table)

#basket analysis
@app.route('/basket', methods=['GET', 'POST'])
def market():
    '''def encode_units(x):
        if x<=0:
            return 0
        if x>=1:
            return 1

    region1 = 'Australia'
    dt = pd.read_excel("datasets/Online Retail.xlsx")
    #print(dt.head())
    dt.columns = dt.columns.str.strip().str.lower().str.replace(" ","_",regex=True).str.replace('('," ",regex=True).str.replace(')'," ",regex=True)
    #dt.isnull().sum()
    dt['invoiceno'] = dt['invoiceno'].astype('str')
    finaldf = pd.DataFrame()
    finaldf1 = pd.DataFrame()
    #for ctry in region1:
    #print(ctry)
    temp = dt[dt['country']== region1].copy()
    #print(temp['country'].unique())
    basket1 = temp.groupby(['invoiceno','description'])['quantity'].sum().unstack().reset_index().fillna(0).set_index('invoiceno')
    #print(basket1.head())
    basket_sets = basket1.applymap(encode_units)
    if 'POSTAGE' in basket_sets.columns:
        basket_sets.drop('POSTAGE',inplace=True,axis=1)
    #frequency
    frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)
    #print (frequent_itemsets)

    #rules
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    #print(rules.head())
    #print("no. of associations = ", rules.shape[0])
    rules['Region']=np.repeat(region1,len(rules.index))
    #rules.hist('confidence',grid = False , bins =30)
    #plt.title('confidence')
    #rules.hist('lift',grid = False , bins =30)
    #plt.title('lift')
    #print("------------reulst-----------")
    #print(rules[ (rules['lift']>=liftt) & (rules['confidence'] >=confit) ])
    finaldf = finaldf.append(rules,ignore_index=True)
    finaldf.to_csv("datasets/results/Association_result_for_online_reatil_data.csv",index=False)
    print("Successful")'''

    if request.method == 'POST':
        form_items = request.form.getlist('items')
        basket.update(form_items)

    recommendations = set()
    basli=[]
    #basli1 ='ALARM CLOCK BAKELIKE GREEN'
    if basket:
        for pro in basket:
            basli.append("frozenset({'"+pro+"'})")
    #data = pd.read_csv("datasets/results/Association_result_for_online_reatil_data.csv")
    data=df[['antecedents', 'consequents']]

    data_new = data[data['antecedents'].isin(basli)]
    finallst = list(data_new['consequents'])
    final =[]
    for each in finallst:
        p1 = each[12:][:-3]
        if "," in p1:
            lss = p1.split(", ")
            for every in lss:
                if "'" in every:
                    re_in = every.index("'")
                    every =  every[:re_in] + every[re_in+1:]
                    every.replace("'","")
                final.append(every)
        else:    
            final.append(p1)
    print(list(set(final)))
    recommendations = list(set(final))

    context = {
        'itemset_count': itemset_count,
        'rules_count': rules_count,
        'items': items,
        'basket': basket,
        'recommendations': recommendations,
    }
    print(recommendations)
    return render_template('market.html', **context)

@app.route('/reset-basket/', methods=['POST'])
def reset_basket():
    global basket
    basket = set()
    return redirect('/basket')

if __name__ == '__main__':
   app.run(debug = True)
