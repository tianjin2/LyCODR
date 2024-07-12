import secrets
import string
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

length = 20
# Construct account classes for accounts used for transactions
class Account(object):
    def __init__(self, address=None, publickey=None):
        self.address = address
        self.publickey = publickey
        self.set_address(self.generation_address())
        self.set_publickey(self.generation_publickey())
    def get_address(self):
        return self.address

    def set_address(self, address):
        self.address = address

    def get_publickey(self):
        return self.publickey

    def set_publickey(self, publickey):
        self.publickey = publickey


    def generation_address(self):
        random_address = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(length))
        return random_address

    def publickey_bytes(self, public_key):
        pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

    def generation_publickey(self):
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        return private_key.public_key()
if __name__ == '__main__':
    account = Account()
    print(account.address)
    print(account.publickey)