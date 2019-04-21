from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, Length

class InputForm(FlaskForm):
	text_description=StringField('Text description', validators=[DataRequired()])
	submit=SubmitField('Generate Image')
	 