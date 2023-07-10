# %%
import numpy as np; import pandas as pd; import matplotlib as mpl; import matplotlib.pyplot as plt; import seaborn as sns # data science lib
# from IPython.core.interactiveshell import InteractiveShell; InteractiveShell.ast_node_interactivity = "all"
import time

# https://ezgmail.readthedocs.io/en/latest/

# There are two packages to send emails: ezgmail and XXX
# for ezgmail, we can't use interactive python! We must use script mode...
# if our .txt file is empty, the attachment will disappear...
# to send to multiple people, the package uses `,` instead of `;` !
# %%
import ezgmail, os
# os.chdir(r'D:\Documents\python\my_notes\email') # the directory where `credentials.json` is located
os.chdir(r'/Users/whlin/Library/CloudStorage/OneDrive-HKUSTConnect/Documents/python/my_notes_py/misc_code/email') # the directory where `credentials.json` is located
# %%
ezgmail.init() # connect to server. Browser will pop up to ask for login. 
# %%
#----------example: send individual emails to multiple people with personalized attachment each----------

grades=['A','B','C']
names=['Tom','mary','Sally']
emails=['lkj872001@gmail.com','whlin@hkcc-polyu.edu.hk', 'whlinaa@connect.ust.hk']
attachments=['sample_1.txt','sample_2.txt','sample_3.txt']
# data=pd.DataFrame(dict(grade=grade, name=name, email=email))
subject='Your grade'
body= """Dear {},

I am very happy to inform you that your grade is {}.

Regards,
WingHo"""

for grade, name, email, attachment in zip(grades, names, emails, attachments):
	ezgmail.send(recipient=email, subject=subject, body=body.format(name,grade), attachments=attachment)
	# ezgmail.send(recipient=email, subject=subject, body=body.format(name,grade), attachments=[attachment])

# another way; using df 
# for i in data.index:
	# ezgmail.send(data['email'][i], subject, body.format(data['name'][i],data['grade'][i])  )	

# %%
#----------example: send an email to multiple people at the same time, with cc----------

body="""Lovely colleague,

Pls disregard this message. This is testing number {}. 
Like our lovely FMO, I'm upgrading my computer and want to test it out. 
I'm sure you'll forgive my testing, like how you forgave FMO... 

If you have any questions, you are VERY WELCOME to contact FMO :>

Finally, let's solve a math Q. What is the A*B, where 

A=\n{}
and 
B=\n{}
?
The answer is:\n{}

"""

email='kwanfrank713@gmail.com'
# email='ricky.mak@cpce-polyu.edu.hk'
# emails=['lkj872001@gmail.com','whlin@hkcc-polyu.edu.hk']
# cc=['whlinaa@connect.ust.hk']
subject='FMO-like testing (number {})'

for i in range(1,20): # send multiple times 
	A,B=np.random.randint(100,size=(5,5)), np.random.randint(100,size=(5,5))
	ans=A@B
	ezgmail.send(email, subject.format(i), body.format(i,A,B,ans))
	# ezgmail.send(','.join(emails), subject.format(i), body.format(i,A,B,ans), cc=','.join(cc))
	# time.sleep(0.8)
# %%

# ','.join(email)


# %%
# another package
# %%
# import smtplib
# smtpObj = smtplib.SMTP('smtp.gmail.com')
# smtpObj.ehlo()
# smtpObj.starttls()
# %%
import smtplib

gmail_user = 'whlinaa@gmail.com'
gmail_password =  # give password here


sent_from = gmail_user
to = ['lkj872001@gmail.com']
subject="hhh"
# to = ['lkj872001@gmail.com', 'whlinaa@connect.ust.hk']
# msg = f'{tips}'
# msg = f'Subject: hihi\n{tips}'
msg = f"""\
From:{gmail_user}
To:{to}
Subject:{subject}

gekko
"""

# print(msg)

server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
server.ehlo()
server.login(gmail_user, gmail_password)
# server.sendmail(sent_from, to, msg)
# server.sendmail(sent_from, to, email_text)
# server.close()
server.quit()
print("GD!")
# %%
msg = f"""\
From:{gmail_user}
To:{to}
Subject:{subject}

gekko
"""
msg