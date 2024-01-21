# from flask import Flask, render_template, request, redirect, url_for, session
# from flask_mysqldb import MySQL
# import bcrypt
#
# app = Flask(__name__)
#
# # Cấu hình kết nối MySQL
# app.config['MYSQL_HOST'] = '127.0.0.1'
# app.config['MYSQL_USER'] = 'root'
# app.config['MYSQL_PASSWORD'] = ''
# app.config['MYSQL_DB'] = 'flask_db'
#
# mysql = MySQL(app)
#
# # Cấu hình secret key để sử dụng session
# app.secret_key = 'your_secret_key'
#
# @app.route('/')
# def index():
#     return render_template('index.html')
#
# @app.route('/register', methods=['GET', 'POST'])
# def register():
#     if request.method == 'POST':
#         username = request.form['username']
#         password = request.form['password'].encode('utf-8')
#         hashed_password = bcrypt.hashpw(password, bcrypt.gensalt())
#
#         cur = mysql.connection.cursor()
#         cur.execute("INSERT INTO admin (username, password) VALUES (%s, %s)", (username, hashed_password))
#         mysql.connection.commit()
#         cur.close()
#
#         return redirect(url_for('login'))
#
#     return render_template('register.html')
#
# @app.route('/login', methods=['POST', 'GET'])
# def login():
#     if request.method == 'POST':
#         username = request.form['username']
#         password_candidate = request.form['password']
#
#         cur = mysql.connection.cursor()
#         result = cur.execute("SELECT * FROM admin WHERE username = %s", [username])
#
#         if result > 0:
#             data = cur.fetchone()
#             password = data[2]  # Thay đổi chỉ số tương ứng với cột 'password'
#
#             if bcrypt.checkpw(password_candidate.encode('utf-8'), password.encode('utf-8')):
#                 session['logged_in'] = True
#                 session['username'] = username
#                 return redirect(url_for('index'))
#             else:
#                 return 'Sai mật khẩu'
#         else:
#             return 'Tài khoản không tồn tại'
#         cur.close()
#
#     return render_template('login.html')
#
# @app.route('/dashboard')
# def dashboard():
#     if 'logged_in' in session:
#         return f'Chào mừng {session["username"]}! Bạn đã đăng nhập thành công.'
#     else:
#         return redirect(url_for('login'))
#
# @app.route('/logout')
# def logout():
#     session.clear()
#     return redirect(url_for('index'))
#
# if __name__ == '__main__':
#     app.run(debug=True)
