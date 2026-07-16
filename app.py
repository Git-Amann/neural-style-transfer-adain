import os
import torch
from flask import Flask, render_template, request, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from PIL import Image
from torchvision import transforms

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization


app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
Bootstrap(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


class UploadForm(FlaskForm):
    content = FileField('Content Image')
    style = FileField('Style Image')
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField('Alpha', default=1.0)
    submit = SubmitField('Transfer Style')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VGG_PATH = 'vgg_normalised.pth'
DECODER_PATH = 'experiment/small_run/decoder_4.pth'   # update once you have a better checkpoint

encoder = VGGEncoder(VGG_PATH).to(device)
decoder = Decoder().to(device)
decoder.load_state_dict(torch.load(DECODER_PATH, map_location=device))

encoder.eval()
decoder.eval()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def style_transfer(content_image, style_image, encoder, decoder, alpha, device):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),   # match training size
        transforms.ToTensor()
    ])

    content_tensor = transform(content_image).unsqueeze(0).to(device)
    style_tensor = transform(style_image).unsqueeze(0).to(device)

    with torch.no_grad():
        content_feat = encoder(content_tensor, is_test=True)   # single tensor (h4)
        style_feat = encoder(style_tensor, is_test=True)       # single tensor (h4)

        t = adaptive_instance_normalization(content_feat, style_feat)
        t = alpha * t + (1 - alpha) * content_feat

        stylized_image = decoder(t)

    return stylized_image


def save_image(image, path):
    image = image.cpu().clone().squeeze(0).clamp(0, 1)
    image = transforms.ToPILImage()(image)
    image.save(path)


@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if form.validate_on_submit():
        if form.content.data and form.content.data.filename:
            if allowed_file(form.content.data.filename):
                content_filename = secure_filename(form.content.data.filename)
                form.content.data.save(os.path.join(app.config['UPLOAD_FOLDER'], content_filename))
        else:
            content_filename = form.content_path.data

        if form.style.data and form.style.data.filename:
            if allowed_file(form.style.data.filename):
                style_filename = secure_filename(form.style.data.filename)
                form.style.data.save(os.path.join(app.config['UPLOAD_FOLDER'], style_filename))
        else:
            style_filename = form.style_path.data

        if content_filename and style_filename:
            content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
            style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)

            try:
                content_image = Image.open(content_path).convert('RGB')
                style_image = Image.open(style_path).convert('RGB')

                alpha = float(form.alpha.data)
                stylized_image = style_transfer(content_image, style_image, encoder, decoder, alpha, device)

                result_filename = 'stylized_' + content_filename
                result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
                save_image(stylized_image, result_path)

                result_image = result_filename
            except Exception as e:
                error = str(e)
    else:
        if not content_filename:
            error = 'Please upload content image'
        if not style_filename:
            error = 'Please upload style image'

    return render_template('index.html', form=form, result_image=result_image,
                           content_image=content_filename, style_image=style_filename, error=error)


@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/examples/<path:filename>')
def send_example(filename):
    return send_from_directory('examples', filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)