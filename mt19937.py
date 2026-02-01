"""
MT19937 (Mersenne Twister) Implementation
Provides access to internal states before and after tempering
"""

class MT19937:
    """
    Mersenne Twister MT19937 implementation
    Exposes internal states for machine learning training
    """

    # MT19937 parameters
    W = 32  # word size (bits)
    N = 624  # degree of recurrence
    M = 397  # middle word offset
    R = 31  # separation point of one word
    A = 0x9908B0DF  # twist matrix parameter

    # Tempering parameters
    U = 11
    D = 0xFFFFFFFF
    S = 7
    B = 0x9D2C5680
    T = 15
    C = 0xEFC60000
    L = 18

    LOWER_MASK = (1 << R) - 1  # 0x7FFFFFFF
    UPPER_MASK = (~LOWER_MASK) & 0xFFFFFFFF  # 0x80000000

    def __init__(self, seed=5489):
        """
        Initialize MT19937 with a seed
        :param seed: initialization seed (default: 5489)
        """
        self.mt = [0] * self.N  # state vector
        self.index = self.N + 1  # index into state vector
        self.seed(seed)

    def seed(self, s):
        """
        Initialize the generator from a seed
        :param s: seed value
        """
        self.mt[0] = s & 0xFFFFFFFF
        for i in range(1, self.N):
            self.mt[i] = (0x6C078965 * (self.mt[i-1] ^ (self.mt[i-1] >> 30)) + i) & 0xFFFFFFFF
        self.index = self.N

    def _twist(self):
        """
        Generate the next N values from the series
        This is the core MT19937 state transition (twisting)
        """
        for i in range(self.N):
            x = (self.mt[i] & self.UPPER_MASK) | (self.mt[(i + 1) % self.N] & self.LOWER_MASK)
            xA = x >> 1
            if x & 1:
                xA ^= self.A
            self.mt[i] = self.mt[(i + self.M) % self.N] ^ xA
        self.index = 0

    def _temper(self, y):
        """
        Tempering function: transforms internal state to output
        :param y: internal state value
        :return: tempered output value
        """
        y ^= (y >> self.U) & self.D
        y ^= (y << self.S) & self.B
        y ^= (y << self.T) & self.C
        y ^= y >> self.L
        return y & 0xFFFFFFFF

    def extract_number(self):
        """
        Extract a tempered value (standard MT19937 output)
        :return: 32-bit random number
        """
        if self.index >= self.N:
            self._twist()

        y = self.mt[self.index]
        y_tempered = self._temper(y)
        self.index += 1

        return y_tempered

    def extract_with_internal(self):
        """
        Extract both internal state and tempered output
        :return: (internal_state, tempered_output) tuple
        """
        if self.index >= self.N:
            self._twist()

        y_internal = self.mt[self.index]
        y_tempered = self._temper(y_internal)
        self.index += 1

        return y_internal, y_tempered

    def get_state(self):
        """
        Get current internal state array
        :return: copy of current state
        """
        return self.mt.copy()

    def set_state(self, state, index=None):
        """
        Set internal state
        :param state: state array (624 values)
        :param index: optional index value
        """
        if len(state) != self.N:
            raise ValueError(f"State must have {self.N} elements")
        self.mt = state.copy()
        if index is not None:
            self.index = index

    def generate_sequence(self, n):
        """
        Generate n random numbers
        :param n: number of values to generate
        :return: list of n random numbers
        """
        return [self.extract_number() for _ in range(n)]

    def generate_with_states(self, n):
        """
        Generate n random numbers with internal states
        :param n: number of values to generate
        :return: (internal_states, tempered_outputs) tuple of lists
        """
        internals = []
        outputs = []

        for _ in range(n):
            internal, tempered = self.extract_with_internal()
            internals.append(internal)
            outputs.append(tempered)

        return internals, outputs


def int_to_bits(value, num_bits=32):
    """
    Convert integer to bit array
    :param value: integer value
    :param num_bits: number of bits (default 32)
    :return: list of bits [b0, b1, ..., b31] (LSB first)
    """
    return [(value >> i) & 1 for i in range(num_bits)]


def bits_to_int(bits):
    """
    Convert bit array to integer
    :param bits: list of bits (LSB first)
    :return: integer value
    """
    return sum(bit << i for i, bit in enumerate(bits))


if __name__ == "__main__":
    # Test MT19937 implementation
    print("Testing MT19937 implementation...")

    mt = MT19937(seed=5489)

    # Generate first 10 numbers
    print("\nFirst 10 numbers:")
    for i in range(10):
        internal, output = mt.extract_with_internal()
        print(f"{i}: Internal={internal:010} ({internal:032b})")
        print(f"   Output  ={output:010} ({output:032b})")

    # Test bit conversion
    print("\nTesting bit conversion:")
    test_val = 0xDEADBEEF
    bits = int_to_bits(test_val)
    recovered = bits_to_int(bits)
    print(f"Original: {test_val:08X}")
    print(f"Bits: {bits[:8]}...")
    print(f"Recovered: {recovered:08X}")
    print(f"Match: {test_val == recovered}")

    # Compare with reference values (first 1000 numbers from MT19937 with seed 5489)
    print("\nComparing first 5 values with reference MT19937:")
    mt2 = MT19937(seed=5489)
    reference = [3499211612, 581869302, 3890346734, 3586334585, 545404204]
    for i, ref in enumerate(reference):
        val = mt2.extract_number()
        match = "OK" if val == ref else "FAIL"
        print(f"{i}: Generated={val:010}, Reference={ref:010} {match}")