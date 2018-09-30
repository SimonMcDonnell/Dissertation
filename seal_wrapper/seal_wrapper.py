import seal
from seal import EncryptionParameters, SEALContext, KeyGenerator, Evaluator, Encryptor, Decryptor, FractionalEncoder, Plaintext, Ciphertext


def seal_obj():
    # params obj
    params = EncryptionParameters()
    # set params
    params.set_poly_modulus("1x^2048 + 1")
    params.set_coeff_modulus(seal.coeff_modulus_128(2048))
    params.set_plain_modulus(1 << 8)
    # get context
    context = SEALContext(params)
    # get evaluator
    evaluator = Evaluator(context)
    # gen keys
    keygen = KeyGenerator(context)
    public_key = keygen.public_key()
    private_key = keygen.secret_key()
    # get encryptor and decryptor
    encryptor = Encryptor(context, public_key)
    decryptor = Decryptor(context, private_key)
    # float number encoder
    encoder = FractionalEncoder(context.plain_modulus(), context.poly_modulus(), 64, 32, 3)
    return evaluator, encoder.encode, encoder.decode, encryptor.encrypt, decryptor.decrypt


evaluate, encode, decode, encrypt, decrypt = seal_obj()


class EA(object):
    
    def __init__(self, values, to_encrypt=False):
        # values must be a list of floats
        self.n = len(values)
        self.encoded_values = []
        self.encrypted_values = []
        
        for i in range(self.n):
            self.encoded_values.append(encode(values[i]))
            
        if to_encrypt:
            for i in range(self.n):
                self.encrypted_values.append(Ciphertext())
                encrypt(self.encoded_values[i], self.encrypted_values[i])
    
    def __getitem__(self, key):
        if self.encrypted_values: 
            v = self.encrypted_values[key]
        else:
            v = self.encoded_values[key]
        return v
    
    def __matmul__(self, other):
        x_mul_w = []
        for i in range(self.n):
            x_mul_w.append(Ciphertext())
            evaluate.multiply_plain(self[i], other[i], x_mul_w[i])
        result = Ciphertext()
        evaluate.add_many(x_mul_w, result)
        v = EA([])
        v.n = 1
        v.encrypted_values.append(result)
        return v
    
    def values(self):
        plain_result = []
        if self.encrypted_values: 
            for i in range(self.n):
                plain_result.append(Plaintext())
                decrypt(self.encrypted_values[i], plain_result[i])
        if not plain_result:
            plain_result = self.encoded_values
        result = [decode(i) for i in plain_result]
        if len(result)==1:
            v = result[0]
        else:
            v = result
        return v
