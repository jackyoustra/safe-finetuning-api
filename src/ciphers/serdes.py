from pathlib import Path

async def get_cipher_from_params(cipher_params):
    cipher_name = cipher_params["cipher_name"]
    cipher_params = cipher_params["cipher_params"]
    return await get_cipher(cipher_name, **cipher_params)

async def get_cipher(class_name, **kwargs):
    import inspect
    import importlib
    import pkgutil

    package_name = "ciphers"
    package = importlib.import_module(package_name)

    CipherClass = None
    # Iterate over all modules in the package
    for _, module_name, _ in pkgutil.iter_modules(package.__path__, prefix=package_name + "."):
        module = importlib.import_module(module_name)
        # Check if the module has the class
        if hasattr(module, class_name):
            CipherClass = getattr(module, class_name)
            break

    # If the class is not found in any module, raise an error
    if not CipherClass:
        raise ImportError(f"Class {class_name} not found in {package_name}")

    def validate_params(func, kwargs):
        sig = inspect.signature(func)
        expected_params = [
            param.name for param in sig.parameters.values()
            if param.name != 'self' and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.VAR_KEYWORD)
        ]

        # Check for unexpected parameters
        for kwarg in kwargs:
            if kwarg not in expected_params and '**' not in str(sig):
                raise ValueError(f"Invalid parameter '{kwarg}' for {func.__name__} in cipher '{class_name}'. Expected parameters: {expected_params}")

        # Check for missing required parameters
        required_params = [
            param.name for param in sig.parameters.values()
            if param.name != 'self' and param.default == inspect.Parameter.empty and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        ]
        missing_params = [param for param in required_params if param not in kwargs]
        if missing_params:
            raise ValueError(f"Missing required parameters for {func.__name__} in cipher '{class_name}': {missing_params}")

    # Check if the class has a 'create' class method
    if hasattr(CipherClass, 'create') and inspect.ismethod(CipherClass.create):
        # Validate kwargs against the create method's parameters
        validate_params(CipherClass.create, kwargs)
        # Use the 'create' method if it exists
        return await CipherClass.create(**kwargs)
    else:
        # Validate kwargs against the CipherClass's __init__ parameters
        validate_params(CipherClass.__init__, kwargs)
        # Initialize the cipher with validated kwargs
        return CipherClass(**kwargs)

def parse_cipher_name(cipher_rel_path: Path):
    """
    Parses cipher relative path to extract class name and parameters.
    Assumes the path is structured as:
    CipherClassName/
        param_key1_value1/
        param_key2_value2/
        ...
    """
    parts = cipher_rel_path.parts  # Tuple of path components
    if not parts:
        raise ValueError(f"Invalid cipher path: {cipher_rel_path}")
    cipher_class_name = parts[0]
    cipher_params = {}
    for part in parts[1:]:
        if part == "default":
            continue  # No parameters to parse

        tokens = part.split('_')
        if len(tokens) < 2:
            # Not enough tokens to have a key and value
            cipher_params[part] = True
            continue

        # Try to parse the last token as a number
        value_str = tokens[-1]
        key_tokens = tokens[:-1]
        key = '_'.join(key_tokens)

        try:
            value = int(value_str)
        except ValueError:
            try:
                value = float(value_str)
            except ValueError:
                # Not a number, treat entire part as key with True value
                key = part
                cipher_params[key] = True
                continue  # Move to next part

        cipher_params[key] = value
    return cipher_class_name, cipher_params

async def get_cipher_from_path(cipher_out_path: Path):
    cipher_class_name, cipher_params = parse_cipher_name(cipher_out_path)
    if cipher_out_path and (cipher_out_path / "cipher_params.json").exists():
        import json
        with open(cipher_out_path / "cipher_params.json", "r") as f:
            cipher_params = json.load(f)
            cipher = await get_cipher_from_params(cipher_params)
    else:
        raise ValueError(f"No cipher params found for {cipher_out_path}")
        # Hacky manual inverse map for now
        # cipher_name_to_class_name = {
        #     "EndSpeak": "EndSpeakCipher",
        # }

        # cipher_class_name = cipher_name_to_class_name.get(cipher_name_in_info, cipher_name_in_info)

        # # Initialize the cipher
        # cipher = await get_cipher(cipher_class_name, **cipher_params)
        # log.info(f"Using cipher: {cipher_class_name} with params: {cipher_params}")
    
    return cipher, cipher_class_name, cipher_params
