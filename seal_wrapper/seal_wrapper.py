import seal
from seal import EncryptionParameters, SEALContext, KeyGenerator, Evaluator, Encryptor, Decryptor, FractionalEncoder, Plaintext, Ciphertext
import numpy as np

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
        # values must be a numpy array of floats
        self.shape = values.shape
        self.encoded_values = []
        self.encrypted_values = []
        self.is_vector = values.ndim == 1

        if self.is_vector:
            for i in range(self.shape[0]):
                self.encoded_values.append(encode(values[i]))
            
            if to_encrypt:
                for i in range(self.shape[0]):
                    self.encrypted_values.append(Ciphertext())
                    encrypt(self.encoded_values[i], self.encrypted_values[i])
        else:
            for i in range(self.shape[0]):
                encoded_row = []
                for j in range(self.shape[1]):
                    encoded_row.append(encode(values[i, j]))
                self.encoded_values.append(encoded_row)
            
            if to_encrypt:
                for i in range(self.shape[0]):
                    encrypted_row = []
                    for j in range(self.shape[1]):
                        encrypted_row.append(Ciphertext())
                        encrypt(self.encoded_values[i][j], encrypted_row[j])
                    self.encrypted_values.append(encrypted_row)
    

    def _getitem(self, key):
        k1, k2 = key
        if self.encrypted_values:
            if self.is_vector:
                v = self.encrypted_values[k1]
            else:
                v = self.encrypted_values[k1][k2]
        else:
            if self.is_vector:
                v = self.encoded_values[k1]
            else:
                v = self.encoded_values[k1][k2]
        return v
    
    def vector_mul(self, other):
        x_mul_w = []
        for i in range(self.shape[0]):
            x_mul_w.append(Ciphertext())
            evaluate.multiply_plain(self._getitem((i,None)), other._getitem((i, None)), x_mul_w[i])
        result = Ciphertext()
        evaluate.add_many(x_mul_w, result)
        v = EA(np.array([]))
        v.shape = (1, 1)
        v.encrypted_values.append(result)
        return v

    def __matmul__(self, other):
        x_mul_w = []
        for x_row in range(self.shape[0]):
            result_row = []
            for w_col in range(other.shape[1]):
                x_row_k_col = []
                for x_col in range(self.shape[1]):
                    x_row_k_col.append(Ciphertext())
                    evaluate.multiply_plain(self._getitem((x_row, x_col)), 
                                            other._getitem((x_col, w_col)), x_row_k_col[x_col])
                result = Ciphertext()
                evaluate.add_many(x_row_k_col, result)
                result_row.append(result)
            x_mul_w.append(result_row)
        v = EA(np.array([]))
        v.shape = (len(x_mul_w), len(x_mul_w[0]))
        v.is_vector = False
        v.encrypted_values = x_mul_w
        return v
        
    def values(self):
        if self.is_vector:
            plain_result = []
            if self.encrypted_values: 
                for i in range(self.shape[0]):
                    plain_result.append(Plaintext())
                    decrypt(self.encrypted_values[i], plain_result[i])
            if not plain_result:
                plain_result = self.encoded_values
            result = [decode(i) for i in plain_result]
            if len(result)==1:
                v = result[0]
            else:
                v = np.array(result)
            return v
        else:
            plain_result = []
            if self.encrypted_values:
                for i in range(self.shape[0]):
                    plain_row = []
                    for j in range(self.shape[1]):
                        plain_row.append(Plaintext())
                        decrypt(self.encrypted_values[i][j], plain_row[j])
                    result = [decode(k) for k in plain_row]
                    plain_result.append(result)
            else:
                for i in range(self.shape[0]):
                    plain_row = self.encoded_values[i]
                    result = [decode(k) for k in plain_row]
                    plain_result.append(result)
            return np.array(plain_result)
