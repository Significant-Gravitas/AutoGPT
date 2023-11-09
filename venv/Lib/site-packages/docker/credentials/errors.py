class StoreError(RuntimeError):
    pass


class CredentialsNotFound(StoreError):
    pass


class InitializationError(StoreError):
    pass


def process_store_error(cpe, program):
    message = cpe.output.decode('utf-8')
    if 'credentials not found in native keychain' in message:
        return CredentialsNotFound(
            'No matching credentials in {}'.format(
                program
            )
        )
    return StoreError(
        'Credentials store {} exited with "{}".'.format(
            program, cpe.output.decode('utf-8').strip()
        )
    )
