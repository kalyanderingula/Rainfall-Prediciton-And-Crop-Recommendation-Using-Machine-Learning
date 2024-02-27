import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from flask import *
import mysql.connector
db=mysql.connector.connect(user="root",password="root",port='3306',database='mydatabase')
cur=db.cursor()

app=Flask(__name__)
app.secret_key="CBJcb786874wrf78chdchsdcv"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/userhome')
def userhome():
    return render_template('userhome.html')

@app.route('/crop')
def crop():
    return render_template('crop.html')

@app.route('/rain')
def rain():
    return render_template('rain.html')

@app.route('/login',methods=['POST','GET'])
def login():
    if request.method=='POST':
        useremail=request.form['useremail']
        session['useremail']=useremail
        userpassword=request.form['userpassword']
        sql="select * from user where useremail='%s' and userpassword='%s'"%(useremail,userpassword)
        cur.execute(sql)
        data=cur.fetchall()
        db.commit()
        if data ==[]:
            msg="user Credentials Are not valid"
            return render_template("login.html",name=msg)
        else:
            return render_template("userhome.html",myname=data[0][0])
    return render_template('login.html')
@app.route('/registration',methods=["POST","GET"])
def registration():
    if request.method=='POST':
        username=request.form['username']
        useremail = request.form['useremail']
        userpassword = request.form['userpassword']
        conpassword = request.form['conpassword']
        Age = request.form['Age']
        
        contact = request.form['contact']
        if userpassword == conpassword:
            sql="select * from user where useremail='%s' and userpassword='%s'"%(useremail,userpassword)
            cur.execute(sql)
            data=cur.fetchall()
            db.commit()
            print(data)
            if data==[]:
                
                sql = "insert into user(username,useremail,userpassword,conpassword,Age,contact)values(%s,%s,%s,%s,%s,%s)"
                val=(username,useremail,userpassword,conpassword,Age,contact)
                cur.execute(sql,val)
                db.commit()
                flash("Registered successfully","success")
                return render_template("login.html")
            else:
                flash("Details are invalid","warning")
                return render_template("registration.html")
        else:
            flash("Password doesn't match", "warning")
            return render_template("registration.html")
    return render_template('registration.html')

@app.route('/view')
def view():
    dataset = pd.read_csv('Crop_recommendation.csv')
    dataset = dataset[:1000]
    print(dataset)
    print(dataset.head(2))
    print(dataset.columns)
    return render_template('view.html', columns=dataset.columns.values, rows=dataset.values.tolist())

@app.route('/preprocess', methods=['POST', 'GET'])
def preprocess():
    global x, y, x_train, x_test, y_train, y_test,  hvectorizer,df
    if request.method == "POST":
        size = int(request.form['split'])
        size = size / 100
        from sklearn.preprocessing import LabelEncoder
        le=LabelEncoder()
        df = pd.read_csv('Crop_recommendation.csv')
        df.head()
        df['label'] = le.fit_transform(df['label'])

        df.head()
        df.columns

       # Assigning the value of x and y 
        x = df.drop(['label'], axis=1)
        y = df['label']

        x_train, x_test, y_train, y_test = train_test_split(x,y, stratify=y, test_size=size, random_state=42)


        # describes info about train and test set
        print("Number transactions X_train dataset: ", x_train.shape)
        print("Number transactions y_train dataset: ", y_train.shape)
        print("Number transactions X_test dataset: ", x_test.shape)
        print("Number transactions y_test dataset: ", y_test.shape)

    
        print(x_train,x_test)
        print(y_train)
        print(y_test)

        return render_template('preprocess.html', msg='Data Preprocessed and It Splits Successfully')
    return render_template('preprocess.html')

@app.route('/model', methods=['POST', 'GET'])
def model():
    if request.method == "POST":
        
        print('ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc')
        s = int(request.form['algo'])
        if s == 0:
            return render_template('model.html', msg='Please Choose an Algorithm to Train')
        elif s == 1:
            print('aaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')

            dt = DecisionTreeClassifier()
            dt.fit(x_train,y_train)
            y_pred = dt.predict(x_test)
            ac_dt = accuracy_score(y_test, y_pred)
            ac_dt = ac_dt * 100
            print('aaaaaaaaaaaaaaaaaaaaaaaaa')
            msg = 'The accuracy obtained by Decision Tree Classifier is  ' + str(ac_dt) + str('%')
            return render_template('model.html', msg=msg)
        elif s == 2:

            classifier = RandomForestClassifier()
            classifier.fit(x_train, y_train)
            y_pred  =  classifier.predict(x_test)            
            
            ac_nb = accuracy_score(y_test, y_pred)
            ac_nb = ac_nb * 100
            msg = 'The accuracy obtained by Random Forest Classifier is ' + str(ac_nb) + str('%')
            return render_template('model.html', msg=msg)
        elif s == 3:

            classifier = AdaBoostClassifier()
            classifier.fit(x_train, y_train)
            y_pred  =  classifier.predict(x_test)            
            
            ac_nb = accuracy_score(y_test, y_pred)
            ac_nb = ac_nb * 100
            msg = 'The accuracy obtained by AdaBoost Classifier is ' + str(ac_nb) + str('%')
            return render_template('model.html', msg=msg)
        elif s == 4:

            classifier = XGBClassifier()
            classifier.fit(x_train, y_train)
            y_pred  =  classifier.predict(x_test)            
            
            ac_nb = accuracy_score(y_test, y_pred)
            ac_nb = ac_nb * 100
            msg = 'The accuracy obtained by XGBoost Classifier is ' + str(ac_nb) + str('%')
            return render_template('model.html', msg=msg)
        elif s == 5:

            classifier = LinearSVC()
            classifier.fit(x_train, y_train)
            y_pred  =  classifier.predict(x_test)            
            
            ac_nb = accuracy_score(y_test, y_pred)
            ac_nb = ac_nb * 100
            msg = 'The accuracy obtained by Support Vector Classifier is ' + str(ac_nb) + str('%')
            return render_template('model.html', msg=msg)
       
    return render_template('model.html')

@app.route('/prediction',methods=['POST','GET'])
def prediction():
    global x_train,y_train
    if request.method == "POST":
        f1 = float(request.form['N'])
        print(f1)
        f2 = float(request.form['P'])
        print(f2)
        f3 = float(request.form['K'])
        print(f3)
        f4 = float(request.form['temperature'])
        print(f4)
        f5 = float(request.form['humidity'])
        print(f5)
        f6 = float(request.form['ph'])
        print(f6)
        f7 = float(request.form['rainfall'])
        print(f7)

        li = [[f1,f2,f3,f4,f5,f6,f7]]
        print(li)
        
        logistic = RandomForestClassifier()
        logistic.fit(x_train,y_train)
        
        result =logistic.predict(li)
        result=result[0]
        if result==0:
            msg = 'The Recommended Crop is predicted as Apple'
        elif result ==1:
            msg= 'The Recommended Crop is predicted as Banana'
        elif result ==2:
            msg= 'The Recommended Crop is predicted as Blackgram'
        elif result ==3:
            msg= 'The Recommended Crop is predicted as Chickpea'
        elif result ==4:
            msg= 'The Recommended Crop is predicted as Coconut'
        elif result ==5:
            msg= 'The Recommended Crop is predicted as Coffee'
        elif result ==6:
            msg= 'The Recommended Crop is predicted as Cotton'
        elif result ==7:
            msg= 'The Recommended Crop is predicted as Grapes'
        elif result ==8:
            msg= 'The Recommended Crop is predicted as Jute'
        elif result ==9:
            msg= 'The Recommended Crop is predicted as Kidneybeans'
        elif result ==10:
            msg= 'The Recommended Crop is predicted as Lentil'
        elif result ==11:
            msg= 'The Recommended Crop is predicted as Maize'
        elif result ==12:
            msg= 'The Recommended Crop is predicted as Mango'
        elif result ==13:
            msg= 'The Recommended Crop is predicted as Mothbeans'
        elif result ==14:
            msg= 'The Recommended Crop is predicted as Moongbeans'
        elif result ==15:
            msg= 'The Recommended Crop is predicted as Muskmelon'
        elif result ==16:
            msg= 'The Recommended Crop is predicted as Orange'
        elif result ==17:
            msg= 'The Recommended Crop is predicted as Papaya'
        elif result ==18:
            msg= 'The Recommended Crop is predicted as Pigeonpeas'
        elif result ==19:
            msg= 'The Recommended Crop is predicted as Pomegranate'
        elif result ==20:
            msg= 'The Recommended Crop is predicted as Rice'
        elif result ==21:
            msg= 'The Recommended Crop is predicted as Watermelon'
        
        return render_template('prediction.html',msg=msg)    

    return render_template('prediction.html')

@app.route('/viewdata')
def viewdata():
    dataset = pd.read_csv('preprocessed_data.csv',usecols=['Date','MinTemp','MaxTemp','Rainfall','Evaporation','Sunshine','WindGustDir','WindGustSpeed','WindDir9am','WindDir3pm','WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Cloud9am','Cloud3pm','Temp9am','Temp3pm','RainToday','RISK_MM','RainTomorrow'])
    dataset = dataset[:1000]
    print(dataset)
    print(dataset.head(2))
    print(dataset.columns)
    return render_template('viewdata.html', columns=dataset.columns.values, rows=dataset.values.tolist())

@app.route('/datacleaning', methods=['POST', 'GET'])
def datacleaning():
    global a, b, a_train, a_test, b_train, b_test,  hvectorizer,df
    if request.method == "POST":
        size = int(request.form['split'])
        size = size / 100
        from sklearn.preprocessing import LabelEncoder
        le=LabelEncoder()
        data = pd.read_csv('preprocessed_data.csv')
        data.head()
        

        # Missing Value imputation
        data['MinTemp'] = data['MinTemp'].fillna(data['MinTemp'].mean())
        data['MaxTemp'] = data['MaxTemp'].fillna(data['MaxTemp'].mean())
        data['Rainfall'] = data['Rainfall'].fillna(data['Rainfall'].mean())
        data['Evaporation'] = data['Evaporation'].fillna(data['Evaporation'].mean())
        data['Sunshine'] = data['Sunshine'].fillna(data['Sunshine'].mean())
        data['WindGustSpeed'] = data['WindGustSpeed'].fillna(data['WindGustSpeed'].mean())
        data['WindSpeed9am'] = data['WindSpeed9am'].fillna(data['WindSpeed9am'].mean())
        data['WindSpeed3pm'] = data['WindSpeed3pm'].fillna(data['WindSpeed3pm'].mean())
        data['Humidity9am'] = data['Humidity9am'].fillna(data['Humidity9am'].mean())
        data['Humidity3pm'] = data['Humidity3pm'].fillna(data['Humidity3pm'].mean())
        data['Pressure9am'] = data['Pressure9am'].fillna(data['Pressure9am'].mean())
        data['Pressure3pm'] = data['Pressure3pm'].fillna(data['Pressure3pm'].mean())
        data['Cloud9am'] = data['Cloud9am'].fillna(data['Cloud9am'].mean())
        data['Cloud3pm'] = data['Cloud3pm'].fillna(data['Cloud3pm'].mean())
        data['Temp9am'] = data['Temp9am'].fillna(data['Temp9am'].mean())
        data['Temp3pm'] = data['Temp3pm'].fillna(data['Temp3pm'].mean())
        data['RISK_MM'] = data['RISK_MM'].fillna(data['RISK_MM'].mean())
        data['WindGustDir'] = data['WindGustDir'].fillna(data['WindGustDir'].mode()[0])
        data['WindDir9am'] = data['WindDir9am'].fillna(data['WindDir9am'].mode()[0])
        data['WindDir3pm'] = data['WindDir3pm'].fillna(data['WindDir3pm'].mode()[0])
        data['RainToday'] = data['RainToday'].fillna(data['RainToday'].mode()[0])
        
        data = data.drop(['Date','Location'],axis = 1)
        data['WindGustDir'] = le.fit_transform(data['WindGustDir'])
        data['WindDir9am'] = le.fit_transform(data['WindDir9am'])
        data['WindDir3pm'] = le.fit_transform(data['WindDir3pm'])
        data['RainToday'] = le.fit_transform(data['RainToday'])
        data['RainTomorrow'] = le.fit_transform(data['RainTomorrow'])

       # Assigning the value of x and y 
        a = data.drop(['RainTomorrow'],axis=1)
        b = data['RainTomorrow']

        a_train, a_test, b_train, b_test = train_test_split(a,b, stratify=b, test_size=size, random_state=42)


        # describes info about train and test set
        # print("Number transactions X_train dataset: ", a_train.shape)
        # print("Number transactions y_train dataset: ", b_train.shape)
        # print("Number transactions X_test dataset: ", a_test.shape)
        # print("Number transactions y_test dataset: ", b_test.shape)

    
        print(a_train,a_test)
        print(b_train)
        print(b_test)

        return render_template('datacleaning.html', msg='Data Preprocessed and It Splits Successfully')
    return render_template('datacleaning.html')

@app.route('/modeltraining', methods=['POST', 'GET'])
def modeltraining():
    if request.method == "POST":
        
        print('ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc')
        s = int(request.form['algo'])
        if s == 0:
            return render_template('modeltraining.html', msg='Please Choose an Algorithm to Train')
        elif s == 1:
            print('aaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')

            dt = DecisionTreeClassifier()
            dt.fit(a_train,b_train)
            b_pred = dt.predict(a_test)
            ac_dt = accuracy_score(b_test, b_pred)
            ac_dt = ac_dt * 100
            print('aaaaaaaaaaaaaaaaaaaaaaaaa')
            msg = 'The accuracy obtained by Decision Tree Classifier is  ' + str(ac_dt) + str('%')
            return render_template('modeltraining.html', msg=msg)
        elif s == 2:

            classifier = RandomForestClassifier()
            classifier.fit(a_train, b_train)
            y_pred  =  classifier.predict(a_test)            
            
            ac_nb = accuracy_score(b_test, y_pred)
            ac_nb = ac_nb * 100
            msg = 'The accuracy obtained by Random Forest Classifier is ' + str(ac_nb) + str('%')
            return render_template('modeltraining.html', msg=msg)
        elif s == 3:

            classifier = AdaBoostClassifier()
            classifier.fit(a_train, b_train)
            y_pred  =  classifier.predict(a_test)            
            
            ac_nb = accuracy_score(b_test, y_pred)
            ac_nb = ac_nb * 100
            msg = 'The accuracy obtained by AdaBoost Classifier is ' + str(ac_nb) + str('%')
            return render_template('modeltraining.html', msg=msg)
        elif s == 4:

            classifier = XGBClassifier()
            classifier.fit(a_train, b_train)
            y_pred  =  classifier.predict(a_test)            
            
            ac_nb = accuracy_score(b_test, y_pred)
            ac_nb = ac_nb * 100
            msg = 'The accuracy obtained by XGBoost Classifier is ' + str(ac_nb) + str('%')
            return render_template('modeltraining.html', msg=msg)
        elif s == 5:

            classifier = LinearSVC()
            classifier.fit(a_train, b_train)
            y_pred  =  classifier.predict(a_test)            
            
            ac_nb = accuracy_score(b_test, y_pred)
            ac_nb = ac_nb * 100
            msg = 'The accuracy obtained by Support Vector Classifier is ' + str(ac_nb) + str('%')
            return render_template('modeltraining.html', msg=msg)
       
    return render_template('modeltraining.html')


@app.route('/prediction1',methods=['POST','GET'])
def prediction1():
    global x_train,y_train
    if request.method == "POST":
        f1 = float(request.form['f1'])
        print(f1)
        f2 = float(request.form['f2'])
        print(f2)
        f3 = float(request.form['f3'])
        print(f3)
        f4 = float(request.form['f4'])
        print(f4)
        f5 = float(request.form['f5'])
        print(f5)
        f6 = float(request.form['f6'])
        print(f6)
        f7 = float(request.form['f7'])
        print(f7)
        f8 = float(request.form['f8'])
        print(f8)
        f9 = float(request.form['f9'])
        print(f9)
        f10= float(request.form['f10'])
        print(f10)
        f11 = float(request.form['f11'])
        print(f11)
        f12 = float(request.form['f12'])
        print(f12)
        f13 = float(request.form['f13'])
        print(f13)
        f14 = float(request.form['f14'])
        print(f14)
        f15 = float(request.form['f15'])
        print(f15)
        f16 = float(request.form['f16'])
        print(f16)
        f17 = float(request.form['f17'])
        print(f17)
        f18 = float(request.form['f18'])
        print(f18)
        f19 = float(request.form['f19'])
        print(f19)
        f20= float(request.form['f20'])
        print(f20)
        f21= float(request.form['f21'])
        print(f21)
        

        li = [[f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21]]
        print(li)
        
        logistic = RandomForestClassifier()
        logistic.fit(a_train,b_train)
        
        result =logistic.predict(li)
        result=result[0]
        if result==0:
            msg = 'There is no rain on Tomorrow'
        elif result ==1:
            msg= 'Tomorrow is going to be a Rainy day'
        
        return render_template('prediction1.html',msg=msg)    

    return render_template('prediction1.html')



if __name__=='__main__':
    app.run(debug=True)