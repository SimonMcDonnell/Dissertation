import seal
from seal import EncryptionParameters, SEALContext, KeyGenerator, EvaluationKeys, Evaluator, Encryptor, Decryptor, FractionalEncoder, Plaintext, Ciphertext
import numpy as np

def seal_obj():
    # params obj
    params = EncryptionParameters()
    # set params
    params.set_poly_modulus("1x^4096 + 1")
    params.set_coeff_modulus(seal.coeff_modulus_128(4096))
    params.set_plain_modulus(1 << 16)
    # get context
    context = SEALContext(params)
    # get evaluator
    evaluator = Evaluator(context)
    # gen keys
    keygen = KeyGenerator(context)
    public_key = keygen.public_key()
    private_key = keygen.secret_key()
    # evaluator keys
    ev_keys = EvaluationKeys()
    keygen.generate_evaluation_keys(30, ev_keys)
    # get encryptor and decryptor
    encryptor = Encryptor(context, public_key)
    decryptor = Decryptor(context, private_key)
    # float number encoder
    encoder = FractionalEncoder(context.plain_modulus(), context.poly_modulus(), 64, 32, 3)
    return evaluator, encoder.encode, encoder.decode, encryptor.encrypt, decryptor.decrypt, ev_keys


evaluate, encode, decode, encrypt, decrypt, ev_keys = seal_obj()


class EA(object):
    
    def __init__(self, values, to_encrypt=False):
        # values must be a numpy array
        # shape must be explicit such as (3,1) and NOT (3,)
        self.shape = values.shape
        self.encoded_values = []
        self.encrypted_values = []

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
            v = self.encrypted_values[k1][k2]
        else:
            v = self.encoded_values[k1][k2]
        return v


    def dot(self, other):
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
        v.encrypted_values = x_mul_w
        return v


    def __add__(self, other):
        x_add_b = []
        for x_row in range(self.shape[0]):
            result_row = []
            for x_col in range(self.shape[1]):
                result_row.append(Ciphertext())
                evaluate.add_plain(self._getitem((x_row, x_col)), other._getitem((0, x_col)),
                                    result_row[x_col])
            x_add_b.append(result_row)
        v = EA(np.array([]))
        v.shape = (len(x_add_b), len(x_add_b[0]))
        v.encrypted_values = x_add_b
        return v

    
    def __sub__(self, other):
        x_sub_b = []
        for x_row in range(self.shape[0]):
            result_row = []
            for x_col in range(self.shape[1]):
                result_row.append(Ciphertext())
                evaluate.sub_plain(self._getitem((x_row, x_col)), other._getitem((0, x_col)),
                                    result_row[x_col])
            x_sub_b.append(result_row)
        v = EA(np.array([]))
        v.shape = (len(x_sub_b), len(x_sub_b[0]))
        v.encrypted_values = x_sub_b
        return v
        
        
    def values(self):
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

    
    def relu(self):
        relu = []
        for x_row in range(self.shape[0]):
            result_row = []
            for x_col in range(self.shape[1]):
                # [-5, 5]
                # coeff1, coeff2, coeff3, coeff4, coeff5 = encode(0.293), encode(0.498), encode(0.164), encode(0.000219), encode(-0.00328)
                # [-6, 3]
                # coeff1, coeff2, coeff3, coeff4, coeff5 = encode(0.308), encode(0.465), encode(0.149), encode(0.00827), encode(-0.000987) 
                # [-4, 4]
                coeff1, coeff2, coeff3, coeff4, coeff5 = encode(0.234), encode(0.497), encode(0.205), encode(0.000428), encode(-0.00642)
                term1, term2, term3, term4, term5, term_extra = Ciphertext(), Ciphertext(), Ciphertext(), Ciphertext(), Ciphertext(), Ciphertext()
                encrypt(coeff1, term1)
                evaluate.multiply_plain(self._getitem((x_row, x_col)), coeff2, term2)
                evaluate.exponentiate(self._getitem((x_row, x_col)), 2, ev_keys, term_extra)
                evaluate.multiply_plain(term_extra, coeff3, term3)
                evaluate.exponentiate(self._getitem((x_row, x_col)), 3, ev_keys, term_extra)
                evaluate.multiply_plain(term_extra, coeff4, term4)
                evaluate.exponentiate(self._getitem((x_row, x_col)), 4, ev_keys, term_extra)
                evaluate.multiply_plain(term_extra, coeff5, term5)
                result = Ciphertext()
                evaluate.add_many([term1, term2, term3, term4, term5], result)
                result_row.append(result)
            relu.append(result_row)
        v = EA(np.array([]))
        v.shape = (len(relu), len(relu[0]))
        v.encrypted_values = relu
        return v

    
    def sigmoid(self):
        sigmoid = []
        for x_row in range(self.shape[0]):
            result_row = []
            for x_col in range(self.shape[1]):
                coeff1, coeff2, coeff3, coeff4 = encode(0.499), encode(0.217), encode(0.000224), encode(-0.00660)
                term1, term2, term3, term4, term_extra = Ciphertext(), Ciphertext(), Ciphertext(), Ciphertext(), Ciphertext()
                encrypt(coeff1, term1)
                evaluate.multiply_plain(self._getitem((x_row, x_col)), coeff2, term2)
                evaluate.exponentiate(self._getitem((x_row, x_col)), 2, ev_keys, term_extra)
                evaluate.multiply_plain(term_extra, coeff3, term3)
                evaluate.exponentiate(self._getitem((x_row, x_col)), 3, ev_keys, term_extra)
                evaluate.multiply_plain(term_extra, coeff4, term4)
                result = Ciphertext()
                evaluate.add_many([term1, term2, term3, term4], result)
                result_row.append(result)
            sigmoid.append(result_row)
        v = EA(np.array([]))
        v.shape = (len(sigmoid), len(sigmoid[0]))
        v.encrypted_values = sigmoid
        return v


    def tanh(self):
        tanh = []
        for x_row in range(self.shape[0]):
            result_row = []
            for x_col in range(self.shape[1]):
                coeff1, coeff2, coeff3, coeff4 = encode(-0.00428), encode(0.600), encode(0.00134), encode(-0.0255)
                term1, term2, term3, term4, term_extra = Ciphertext(), Ciphertext(), Ciphertext(), Ciphertext(), Ciphertext()
                encrypt(coeff1, term1)
                evaluate.multiply_plain(self._getitem((x_row, x_col)), coeff2, term2)
                evaluate.exponentiate(self._getitem((x_row, x_col)), 2, ev_keys, term_extra)
                evaluate.multiply_plain(term_extra, coeff3, term3)
                evaluate.exponentiate(self._getitem((x_row, x_col)), 3, ev_keys, term_extra)
                evaluate.multiply_plain(term_extra, coeff4, term4)
                result = Ciphertext()
                evaluate.add_many([term1, term2, term3, term4], result)
                result_row.append(result)
            tanh.append(result_row)
        v = EA(np.array([]))
        v.shape = (len(tanh), len(tanh[0]))
        v.encrypted_values = tanh
        return v