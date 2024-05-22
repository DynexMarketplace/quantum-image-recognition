# Quantum Image Recognition
In this demo we will use the Dynex SDK to perform an image classification task using a Quantum-Restricted-Boltzmann-Machine ("QRBM") based on the paper "A hybrid quantum-classical approach for inference on restricted Boltzmann machines".

# Usage

1. Launch in Github Codespaces and wait until the codepsace is fully initialised

2. Add your account keys by drag&drop of your dynex.ini into the main folder

3. Verify your account keys by typing the following command in the console:

```
python
>>> import dynex
>>> dynex.test()
>>> exit()
```

Your console will perform tests and validate your account keys. You should see the following message:

```
[DYNEX] TEST RESULT: ALL TESTS PASSED
```

4. Run the demo by typing the following command:

```
python main.py
```

The program will output and save the reconstructed images in the file "result.png".


