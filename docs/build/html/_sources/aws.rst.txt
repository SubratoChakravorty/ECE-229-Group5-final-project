.. AWS documentation .

Configuring AWS
===================================

DeepCTR is a **Easy-to-use** , **Modular** and **Extendible** package of deep-learning based CTR models along with lots of core components layer  which can be used to easily build custom models.It is compatible with **tensorflow 1.4+ and 2.0+**.You can use any complex model with ``model.fit()`` and ``model.predict()``.

Amazon Elastic Compute Cloud is a web service that provides secure, resizable compute capacity in the cloud. It is designed to make web-scale cloud computing easier for developers. 



You can access the website at https://www.9thgraders.com


- Configure Virtual Private Cloud

- Configure the EC2 server on AWS

- Install the necessary components on the server
This requires several applications which can be listed as below:
  + Python
  + pip
  + apache2
  + mod_wsgi
  + dash, dash-renderer, dash-html-components 
  + plotly
  + Flask

- Configure the mode_wsgi
run the command ``mod_wsgi-express module-config``, you should see something like: 
LoadModule wsgi_module "/usr/local/lib/python3.6/dist-packages/mod_wsgi/server/mod_wsgi-py36.cpython-36m-x86_64-linux-gnu.so"
WSGIPythonHome "/usr"
Write the output into */etc/apache2/mods-available/wsgi.load*

- Enable the site
Now the site can be enabled by ``sudo a2ensite FlaskApp`` and reload the apache with ``service apache2 reload``.

- Setup WSGI

- Move your project to the project path

Now you should be able to visit the site via IP address after reloading the apache.


