from flask import Flask, render_template, request, session, redirect, url_for, Response, jsonify, send_file, flash
import mysql.connector
import cv2
from PIL import Image
import numpy as np
import os
import time
from datetime import date
from openpyxl import Workbook
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import pandas as pd
app = Flask(__name__)
app.secret_key = 'your_secret_key'
cnt = 0
pause_cnt = 0
justscanned = False

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="",
    database="flask_db"
)
mycursor = mydb.cursor()


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Generate dataset >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def generate_dataset(nbr):
    face_classifier = cv2.CascadeClassifier(
        "C:/Users/ADMIN/PycharmProjects/FlaskOpencv_FaceRecognition/resources/haarcascade_frontalface_default.xml")

    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        # scaling factor=1.3
        # Minimum neighbor = 5

        if faces is ():
            return None
        for (x, y, w, h) in faces:
            cropped_face = img[y:y + h, x:x + w]
        return cropped_face

    cap = cv2.VideoCapture(0)

    mycursor.execute("select ifnull(max(img_id), 0) from img_dataset")
    row = mycursor.fetchone()
    lastid = row[0]

    img_id = lastid
    max_imgid = img_id + 100
    count_img = 0

    while True:
        ret, img = cap.read()
        if face_cropped(img) is not None:
            count_img += 1
            img_id += 1
            face = cv2.resize(face_cropped(img), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            file_name_path = "dataset/" + nbr + "." + str(img_id) + ".jpg"
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(count_img), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            mycursor.execute("""INSERT INTO `img_dataset` (`img_id`, `img_person`) VALUES
                                ('{}', '{}')""".format(img_id, nbr))
            mydb.commit()

            frame = cv2.imencode('.jpg', face)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            if cv2.waitKey(1) == 13 or int(img_id) == int(max_imgid):
                break
                cap.release()
                cv2.destroyAllWindows()


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Train Classifier >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
@app.route('/train_classifier/<nbr>')
def train_classifier(nbr):
    dataset_dir = "C:/Users/ADMIN/PycharmProjects/FlaskOpencv_FaceRecognition/dataset"

    path = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)]
    faces = []
    ids = []

    for image in path:
        img = Image.open(image).convert('L');
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])

        faces.append(imageNp)
        ids.append(id)
    ids = np.array(ids)

    # Train the classifier and save
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")

    return redirect('/')


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Face Recognition >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def face_recognition():  # generate frame by frame from camera
    def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)

        global justscanned
        global pause_cnt

        pause_cnt += 1

        coords = []

        # Đếm số khuôn mặt được phát hiện
        num_faces = len(features)

        for (x, y, w, h) in features:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            id, pred = clf.predict(gray_image[y:y + h, x:x + w])
            confidence = int(100 * (1 - pred / 300))

            if confidence > 70 and not justscanned:
                global cnt
                cnt += 1

                n = (100 / 30) * cnt
                w_filled = (cnt / 30) * w

                cv2.putText(img, str(int(n)) + ' %', (x + 20, y + h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (153, 255, 255), 2, cv2.LINE_AA)

                cv2.rectangle(img, (x, y + h + 40), (x + w, y + h + 50), color, 2)
                cv2.rectangle(img, (x, y + h + 40), (x + int(w_filled), y + h + 50), (153, 255, 255), cv2.FILLED)

                mycursor.execute("select a.img_person, b.prs_name, b.prs_skill "
                                 "  from img_dataset a "
                                 "  left join prs_mstr b on a.img_person = b.prs_nbr "
                                 " where img_id = " + str(id))
                row = mycursor.fetchone()
                pnbr = row[0]
                pname = row[1]
                pskill = row[2]

                if int(cnt) == 30:
                    cnt = 0

                    mycursor.execute("insert into accs_hist (accs_date, accs_prsn) values('" + str(
                        date.today()) + "', '" + pnbr + "')")
                    mydb.commit()

                    cv2.putText(img, pname + ' | ' + pskill, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (153, 255, 255), 2, cv2.LINE_AA)
                    time.sleep(1)

                    justscanned = True
                    pause_cnt = 0

            else:
                if not justscanned:
                    cv2.putText(img, 'UNKNOWN', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(img, ' ', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

                if pause_cnt > 80:
                    justscanned = False

            coords = [x, y, w, h]

        # Hiển thị số khuôn mặt trên video
        cv2.putText(img, "Number of Faces: " + str(num_faces), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return coords

    def recognize(img, clf, faceCascade):
        coords = draw_boundary(img, faceCascade, 1.1, 10, (255, 255, 0), "Face", clf)
        return img

    faceCascade = cv2.CascadeClassifier(
        "C:/Users/ADMIN/PycharmProjects/FlaskOpencv_FaceRecognition/resources/haarcascade_frontalface_default.xml")
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")

    wCam, hCam = 400, 400

    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    while True:
        ret, img = cap.read()
        img = recognize(img, clf, faceCascade)

        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        key = cv2.waitKey(1)
        if key == 27:
            break


@app.route('/')
def default_route():
    return redirect(url_for('login'))

# HÀM NÀY XỬ LÝ BẮT BUỘC PHẢI LOGIN ,NẾU KHÔNG SẼ KHÔNG VÀO ĐƯỢC HỆ THỐNG
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Login required.')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/home')
@login_required
def home():
    mycursor.execute("select prs_nbr, prs_name, prs_skill, prs_active, prs_added from prs_mstr")
    data = mycursor.fetchall()

    return render_template('index.html', data=data)

@app.route('/export_to_excel', methods=['GET'])
def export_to_excel():
    # Lấy dữ liệu từ cơ sở dữ liệu hoặc từ nơi bạn lưu trữ dữ liệu
    mycursor.execute("SELECT * FROM prs_mstr")
    data = mycursor.fetchall()

    # Tạo một Workbook và một Sheet trong tệp Excel
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Face Recognition Data"

    # Điền dữ liệu vào tệp Excel
    sheet.append(["Person Id", "Name", "Skill", "Active", "Added"])
    for item in data:
        sheet.append([item[0], item[1], item[2], item[3], item[4]])

    # Lưu tệp Excel
    excel_filename = "Danh sách người dùng có trong hệ thống.xlsx"
    workbook.save(excel_filename)

    # Trả về tệp Excel cho người dùng để tải xuống
    return send_file(excel_filename, as_attachment=True)

@app.route('/addprsn')
def addprsn():
    mycursor.execute("select ifnull(max(prs_nbr) + 1, 1) from prs_mstr")
    row = mycursor.fetchone()
    nbr = row[0]
    # print(int(nbr))

    return render_template('addprsn.html', newnbr=int(nbr))


@app.route('/addprsn_submit', methods=['POST'])
def addprsn_submit():
    prsnbr = request.form.get('txtnbr')
    prsname = request.form.get('txtname')
    prsskill = request.form.get('optskill')

    mycursor.execute("""INSERT INTO `prs_mstr` (`prs_nbr`, `prs_name`, `prs_skill`) VALUES
                    ('{}', '{}', '{}')""".format(prsnbr, prsname, prsskill))
    mydb.commit()

    # return redirect(url_for('home'))
    return redirect(url_for('vfdataset_page', prs=prsnbr))


@app.route('/vfdataset_page/<prs>')
def vfdataset_page(prs):
    return render_template('gendataset.html', prs=prs)


@app.route('/vidfeed_dataset/<nbr>')
def vidfeed_dataset(nbr):
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(generate_dataset(nbr), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(face_recognition(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/fr_page')
def fr_page():
    """Video streaming home page."""
    mycursor.execute("select a.accs_id, a.accs_prsn, b.prs_name, b.prs_skill, a.accs_added "
                     "  from accs_hist a "
                     "  left join prs_mstr b on a.accs_prsn = b.prs_nbr "
                     " where a.accs_date = curdate() "
                     " order by 1 desc")
    data = mycursor.fetchall()

    return render_template('fr_page.html', data=data)


@app.route('/countTodayScan')
def countTodayScan():
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="",
        database="flask_db"
    )
    mycursor = mydb.cursor()

    mycursor.execute("select count(*) "
                     "  from accs_hist "
                     " where accs_date = curdate() ")
    row = mycursor.fetchone()
    rowcount = row[0]

    return jsonify({'rowcount': rowcount})


@app.route('/loadData', methods=['GET', 'POST'])
def loadData():
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="",
        database="flask_db"
    )
    mycursor = mydb.cursor()

    mycursor.execute("select a.accs_id, a.accs_prsn, b.prs_name, b.prs_skill, date_format(a.accs_added, '%H:%i:%s') "
                     "  from accs_hist a "
                     "  left join prs_mstr b on a.accs_prsn = b.prs_nbr "
                     " where a.accs_date = curdate() "
                     " order by 1 desc")
    data = mycursor.fetchall()

    return jsonify(response=data)

@app.route('/edit_record/<prs>', methods=['GET', 'POST'])
def edit_record(prs):
    if request.method == 'GET':
        # Fetch record details based on prs
        mycursor.execute("SELECT * FROM prs_mstr WHERE prs_nbr = %s", (prs,))
        data = mycursor.fetchone()
        return render_template('edit_record.html', data=data)
    elif request.method == 'POST':
        # Handle form submission to update record
        new_name = request.form.get('new_name')
        new_skill = request.form.get('new_skill')

        mycursor.execute("UPDATE prs_mstr SET prs_name = %s, prs_skill = %s WHERE prs_nbr = %s", (new_name, new_skill, prs))
        mydb.commit()

        return redirect(url_for('home'))

@app.route('/delete_record/<prs>')
def delete_record(prs):
    # Delete record based on prs
    mycursor.execute("DELETE FROM prs_mstr WHERE prs_nbr = %s", (prs,))
    mydb.commit()

    return redirect(url_for('home'))
# Registration Route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        email = request.form.get('email')

        hashed_password = generate_password_hash(password, method='sha256')

        mycursor.execute("INSERT INTO users (username, password, email) VALUES (%s, %s, %s)",
                         (username, hashed_password, email))
        mydb.commit()

        flash('Registration successful. Please log in.')
        return redirect(url_for('login'))

    return render_template('register.html')

# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        mycursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = mycursor.fetchone()

        if user and check_password_hash(user[2], password):
            session['user_id'] = user[0]
            session['username'] = user[1]  # Set the username in the session
            return redirect(url_for('home'))

        flash('Login failed. Check your username and password.')

    return render_template('login.html')

# Logout Route
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))

# Secure Page Route (Example)
@app.route('/secure_page')
def secure_page():
    if 'user_id' in session:
        # This route is accessible only for logged-in users
        return render_template('navbar.html')
    else:
        flash('Login required.')
        return redirect(url_for('login'))
@app.route('/user_list')
def user_list():
    try:
        # Retrieve the user ID from the session
        user_id = session.get('user_id')

        if user_id:
            # Query the database for the information of the logged-in user
            mycursor.execute("SELECT * FROM users WHERE user_id = %s", (user_id,))
            user = mycursor.fetchone()

            # Check if the user exists
            if user:
                return render_template('user_list.html', users=[user])
            else:
                return "User not found."
        else:
            return "User not logged in."

    except Exception as e:
        return f"An error occurred: {e}"

@app.route('/edit_user/<int:user_id>', methods=['GET', 'POST'])
def edit_user(user_id):
    try:
        if request.method == 'POST':
            new_username = request.form.get('new_username')
            new_email = request.form.get('new_email')

            mycursor.execute("UPDATE users SET username = %s, email = %s WHERE user_id = %s", (new_username, new_email, user_id))
            mydb.commit()

            return redirect(url_for('user_list'))

        mycursor.execute("SELECT * FROM users WHERE user_id = %s", (user_id,))
        user = mycursor.fetchone()

        return render_template('edit_user.html', user=user)
    except Exception as e:
        return f"An error occurred: {e}"

@app.route('/delete_user/<int:user_id>')
def delete_user(user_id):
    try:
        mycursor.execute("DELETE FROM users WHERE user_id = %s", (user_id,))
        mydb.commit()

        return redirect(url_for('user_list'))
    except Exception as e:
        return f"An error occurred: {e}"



@app.route('/export_excel')
def export_excel():
    try:
        # Get the user ID from the session (adjust as per your authentication mechanism)
        user_id = session.get('user_id')

        # Check if the user is logged in
        if user_id is None:
            return "User not logged in"

        # Modify the SQL query to fetch user-specific data without the 'id' column
        mycursor.execute("SELECT `username`, `email`, `created_at` FROM users WHERE `user_id` = %s", (user_id,))
        user_data = mycursor.fetchall()

        # Check if user data is found
        if not user_data:
            return "User not found"

        # Check the number of columns in the database result
        num_columns_db = len(user_data[0])

        # Define column names based on the number of columns in the database
        column_names = ['username', 'email', 'created_at'][:num_columns_db]

        # Convert data to DataFrame
        df = pd.DataFrame(user_data, columns=column_names)

        # Create a unique filename (e.g., based on timestamp)
        excel_filename = 'Danh sách thông tin quản trị viên.xlsx'

        # Export DataFrame to Excel file
        df.to_excel(excel_filename, index=False)

        # Return the Excel file as a response
        return send_file(excel_filename, as_attachment=True)

    except Exception as e:
        return f"An error occurred: {e}"
@app.route('/change_password', methods=['GET', 'POST'])
@login_required
def change_password():
    if request.method == 'POST':
        user_id = session['user_id']
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')

        # Validate that new password matches the confirmation
        if new_password != confirm_password:
            flash('New password and confirmation do not match.')
            return redirect(url_for('change_password'))

        # Hash the new password
        hashed_password = generate_password_hash(new_password, method='sha256')

        # Replace 'user_id' with the actual primary key column name in your 'users' table
        mycursor.execute("UPDATE users SET password = %s WHERE user_id = %s", (hashed_password, user_id))
        mydb.commit()

        flash('Password changed successfully.')
        return redirect(url_for('home'))

    return render_template('change_password.html')

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)
