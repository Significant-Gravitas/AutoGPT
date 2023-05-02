sudo nano /etc/httpd/conf/httpd.conf
DocumentRoot "/var/www/html"
<Directory "/var/www/html">
    AllowOverride All
    Require all granted
</Directory>
sudo systemctl restart httpd
