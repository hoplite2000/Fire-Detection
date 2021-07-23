import smtplib
from alert_cred import *

def mailer():
    try:
        smtpObj = smtplib.SMTP('smtp.gmail.com', 587)
        smtpObj.starttls()
        smtpObj.login(user,passw)
        smtpObj.sendmail(sender, receivers, message)
        print ("Successfully sent email")
    except:
        print ("Error!")