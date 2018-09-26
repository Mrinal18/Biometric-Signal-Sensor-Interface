from hexoskin import client, errors
import configparser

__author__ = 'Simar Singh'

class SessionInfo():

    def __init__(self, public_key, private_key, username, password):
        self.api = client.HexoApi(api_key=public_key, api_secret=private_key, api_version='', auth=username + ':' + password, base_url=None)
        authorization_code = self.test_login()
        if authorization_code == 'successful':
            print('Login Successful')
        else:
            print('Login Unsuccessful')

    def test_login(self):
        try:
            user = self.api.account.list()
        except Exception as e:
            return 'unsuccessful'
        return 'successful'

def api_login():
    config = configparser.ConfigParser()
    config.read('config.ini')
    public_key = config.get('LoginInfo', 'public_key')
    private_key = config.get('LoginInfo', 'private_key')
    username = config.get('LoginInfo', 'username')
    password = config.get('LoginInfo', 'password')

    session = SessionInfo(public_key, private_key, username, password)
    return session
