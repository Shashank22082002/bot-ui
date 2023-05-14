
from flask import Flask, request, jsonify, render_template
from model import targetFunction

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route("/")
def home():
    return render_template('index.html')

# لقد نسيت كلمة السر. ماذا يجب أن أفعل؟ 
# I forgot the password. What should I do?
# إذا نسيت كلمة المرور الخاصة بي ، فماذا أفعل؟
# What can I do If I forgot my password?s

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.form['query']
    print(data);
    response = targetFunction(data)
    data = response['RESPONSE']
    return render_template('index.html', response=data)

if __name__ == '__main__':
    app.run(debug=True)