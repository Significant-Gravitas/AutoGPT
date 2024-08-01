

const Marketplace = () => {

    return (

        <div className="relative overflow-hidden bg-white">

            <section aria-labelledby="sale-heading" className="relative mx-auto flex max-w-7xl flex-col items-center px-4 pt-32 mb-10 text-center sm:px-6 lg:px-8">
                <div aria-hidden="true" className="absolute inset-0">
                    <div className="absolute inset-0 mx-auto max-w-7xl overflow-hidden xl:px-8">
                        <img src="https://images.unsplash.com/photo-1562408590-e32931084e23?q=80&w=3270&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D" alt="" className="w-full object-cover object-center" />
                    </div>
                    <div className="absolute inset-0 bg-white bg-opacity-75"></div>
                    <div className="absolute inset-0 bg-gradient-to-t from-white via-white"></div>
                </div>
                <div className="mx-auto max-w-2xl lg:max-w-none relative z-10">
                    <h2 id="sale-heading" className="text-4xl font-bold tracking-tight text-gray-900 sm:text-5xl lg:text-6xl">AutoGPT Marketplace</h2>
                    <p className="mx-auto mt-4 max-w-xl text-xl text-gray-600">Discover and Share proven AI Agents and supercharge your business.</p>
                </div>
            </section>

            <section aria-labelledby="testimonial-heading" className="relative justify-center mx-auto max-w-7xl px-4 sm:px-6 lg:py-8">

                <form className="mb-4 flex justify-center">
                    <input
                        type="text"
                        placeholder="Search..."
                        className="w-3/4 max-w-xl px-3 py-2 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring focus:ring-indigo-200"
                    />
                </form>
                <ul role="list" className="divide-y divide-gray-100">

                    <li className="flex justify-between gap-x-6 py-5">
                        <div className="flex min-w-0 gap-x-4">
                            <img className="h-12 w-12 flex-none rounded-full bg-gray-50" src="https://images.unsplash.com/photo-1562408590-e32931084e23?q=80&w=3270&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D" alt="" />
                            <div className="min-w-0 flex-auto">
                                <p className="text-sm font-semibold leading-6 text-gray-900">
                                    <a href="#">
                                        <span className="absolute inset-x-0 -top-px bottom-0"></span>
                                        Agent Name
                                    </a>
                                </p>
                                <p className="mt-1 flex text-xs leading-5 text-gray-500">
                                    <a href="mailto:leslie.alexander@example.com" className="relative truncate hover:underline">Agent description</a>
                                </p>
                            </div>
                        </div>
                        <div className="flex shrink-0 items-center gap-x-4">
                            <div className="hidden sm:flex sm:flex-col sm:items-end">
                                <p className="text-sm leading-6 text-gray-900">Category</p>
                                <p className="mt-1 text-xs leading-5 text-gray-500">Last updated <time dateTime="2023-01-23T13:23Z">3h ago</time></p>
                            </div>
                            <svg className="h-5 w-5 flex-none text-gray-400" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                                <path fill-rule="evenodd" d="M7.21 14.77a.75.75 0 01.02-1.06L11.168 10 7.23 6.29a.75.75 0 111.04-1.08l4.5 4.25a.75.75 0 010 1.08l-4.5 4.25a.75.75 0 01-1.06-.02z" clip-rule="evenodd" />
                            </svg>
                        </div>
                    </li>
                    <li className="flex justify-between gap-x-6 py-5">
                        <div className="flex min-w-0 gap-x-4">
                            <img className="h-12 w-12 flex-none rounded-full bg-gray-50" src="https://images.unsplash.com/photo-1562408590-e32931084e23?q=80&w=3270&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D" alt="" />
                            <div className="min-w-0 flex-auto">
                                <p className="text-sm font-semibold leading-6 text-gray-900">
                                    <a href="#">
                                        <span className="absolute inset-x-0 -top-px bottom-0"></span>
                                        Agent Name
                                    </a>
                                </p>
                                <p className="mt-1 flex text-xs leading-5 text-gray-500">
                                    <a href="mailto:leslie.alexander@example.com" className="relative truncate hover:underline">Agent description</a>
                                </p>
                            </div>
                        </div>
                        <div className="flex shrink-0 items-center gap-x-4">
                            <div className="hidden sm:flex sm:flex-col sm:items-end">
                                <p className="text-sm leading-6 text-gray-900">Category</p>
                                <p className="mt-1 text-xs leading-5 text-gray-500">Last updated <time dateTime="2023-01-23T13:23Z">3h ago</time></p>
                            </div>
                            <svg className="h-5 w-5 flex-none text-gray-400" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                                <path fill-rule="evenodd" d="M7.21 14.77a.75.75 0 01.02-1.06L11.168 10 7.23 6.29a.75.75 0 111.04-1.08l4.5 4.25a.75.75 0 010 1.08l-4.5 4.25a.75.75 0 01-1.06-.02z" clip-rule="evenodd" />
                            </svg>
                        </div>
                    </li>
                    <li className="flex justify-between gap-x-6 py-5">
                        <div className="flex min-w-0 gap-x-4">
                            <img className="h-12 w-12 flex-none rounded-full bg-gray-50" src="https://images.unsplash.com/photo-1562408590-e32931084e23?q=80&w=3270&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D" alt="" />
                            <div className="min-w-0 flex-auto">
                                <p className="text-sm font-semibold leading-6 text-gray-900">
                                    <a href="#">
                                        <span className="absolute inset-x-0 -top-px bottom-0"></span>
                                        Agent Name
                                    </a>
                                </p>
                                <p className="mt-1 flex text-xs leading-5 text-gray-500">
                                    <a href="mailto:leslie.alexander@example.com" className="relative truncate hover:underline">Agent description</a>
                                </p>
                            </div>
                        </div>
                        <div className="flex shrink-0 items-center gap-x-4">
                            <div className="hidden sm:flex sm:flex-col sm:items-end">
                                <p className="text-sm leading-6 text-gray-900">Category</p>
                                <p className="mt-1 text-xs leading-5 text-gray-500">Last updated <time dateTime="2023-01-23T13:23Z">3h ago</time></p>
                            </div>
                            <svg className="h-5 w-5 flex-none text-gray-400" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                                <path fill-rule="evenodd" d="M7.21 14.77a.75.75 0 01.02-1.06L11.168 10 7.23 6.29a.75.75 0 111.04-1.08l4.5 4.25a.75.75 0 010 1.08l-4.5 4.25a.75.75 0 01-1.06-.02z" clip-rule="evenodd" />
                            </svg>
                        </div>
                    </li>
                </ul>
            </section>
        </div>
    );

};

export default Marketplace;
