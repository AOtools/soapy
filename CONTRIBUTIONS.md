#Contributing to SOAPY
---------------------

Contributing to SOPY is very much encouraged. I haven't an enormous amount of time to add new code and features to SOAPY, so if you can and want to help please do! 

To begin, make yourself a fork using the Github interface, then use the resulting repository as your origin for local git repositories. Once you've implemented a feature or fix, again use the github interface to issue a "pull-request". 

##Style Guidelines
________________

Broadly, it's good if new code follows the [pep8](https://www.python.org/dev/peps/pep-0008/), with a few other additions. In summary:
  - All modules, classes and methods should have decent doc strings, which say what inputs and outputs are
  - Doc-strings should be in the ["google style"](http://sphinxcontrib-napoleon.readthedocs.org/en/latest/example_google.html), so indented ``Parameters``, and then ``Return`` values. There are any examples in the code already.
  - Try to use "new-style" printing, so use ``print`` as a function rather than a statement
  - And "new-style" string formatting, so ``"number is: {}".format(number)``
