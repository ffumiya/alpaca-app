import os
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image

import numpy as np

#分類したいクラスと学習に用いた画像のサイズ
classes = ['アルパカ', 'ヒツジ']
img_size = 64

#アップロードされた画像を保存するフォルダ名とアップロードを許可する拡張子
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

#Flaskクラスのインスタンスの作成
app = Flask(__name__)

#アップロードされたファイルの拡張子のチェックをする関数
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#学習済みモデルをロード
model = load_model('./animal_cnn.h5')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            print('ファイルが選択されました')  # ファイルが選択されたことを確認
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            filepath = os.path.join("uploads", filename)

            # 受け取った画像を読み込み、np形式に変換
            img = image.load_img('static/' + filepath, grayscale=True, target_size=(img_size,img_size))
            img = img.convert('RGB')
            # 画像データを64 x 64に変換
            img = img.resize((img_size, img_size))
            # 画像データをnumpy配列に変換
            img = np.asarray(img)
            img = img / 255.0
            result = model.predict(np.array([img]))
            predicted = result.argmax()
            pred_answer = "この動物は " + classes[predicted] + " です"

            print('レンダリングされる前の処理')  # レンダリングされる前の処理が実行されることを確認
            return render_template("index.html", answer=pred_answer, filepath=filepath)

    return render_template("index.html", answer="")


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0',port = port)