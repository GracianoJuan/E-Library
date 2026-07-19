"use client";

import Link from "next/link";
import { useAuth } from "@/hooks/useAuth";

export default function Header() {
    const { user, isLoggedIn, logout } = useAuth();

    return (
        <header className="border-b border-slate-200 bg-white/90 backdrop-blur supports-backdrop-filter:bg-white/70 dark:border-slate-800 dark:bg-slate-950/90">
            <div className="mx-auto flex max-w-7xl flex-col gap-4 px-4 py-4 sm:px-6 lg:flex-row lg:items-center lg:justify-between lg:gap-6 lg:px-8">
                <div className="flex items-center gap-4">
                    <img src="/logo%20(2).png" alt="E-Library logo" className="h-12 w-12 rounded-md object-contain" />
                    <div>
                        <h1 className="text-xl font-black tracking-tight text-slate-950 sm:text-2xl dark:text-white">E-Library</h1>
                        <p className="text-sm text-slate-500 dark:text-slate-400">Discover your next favorite book</p>
                    </div>
                </div>

                {isLoggedIn && user ? (
                    <div className="flex w-full flex-col gap-3 sm:w-auto sm:flex-row sm:items-center sm:justify-end">
                        <div className="hidden text-right sm:block">
                            <p className="text-sm font-semibold text-slate-900 dark:text-white">{user.name}</p>
                            <p className="text-xs text-slate-500 dark:text-slate-400">{user.email}</p>
                        </div>
                        <button
                            type="button"
                            onClick={logout}
                            className="w-full rounded-full bg-slate-950 px-4 py-2 text-sm font-semibold text-white transition hover:bg-slate-800 sm:w-auto dark:bg-white dark:text-slate-950 dark:hover:bg-slate-200"
                        >
                            Logout
                        </button>
                    </div>
                ) : (
                    <div className="flex w-full flex-col gap-3 sm:w-auto sm:flex-row sm:items-center">
                        <Link
                            href="/login"
                            className="w-full rounded-full border border-slate-300 px-4 py-2 text-center text-sm font-semibold text-slate-700 transition hover:border-slate-950 hover:text-slate-950 sm:w-auto dark:border-slate-700 dark:text-slate-300 dark:hover:border-white dark:hover:text-white"
                        >
                            Login
                        </Link>
                        <Link
                            href="/register"
                            className="w-full rounded-full bg-slate-950 px-4 py-2 text-center text-sm font-semibold text-white transition hover:bg-slate-800 sm:w-auto dark:bg-white dark:text-slate-950 dark:hover:bg-slate-200"
                        >
                            Register
                        </Link>
                    </div>
                )}
            </div>
        </header>
    );
}
